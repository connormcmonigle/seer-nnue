/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <board.h>
#include <history_heuristic.h>
#include <local_cache.h>
#include <move.h>
#include <move_orderer.h>
#include <nnue_model.h>
#include <search_constants.h>
#include <search_stack.h>
#include <syzygy.h>
#include <transposition_table.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <unordered_map>
#include <vector>

namespace search {

template <bool is_root>
struct pv_search_result {};

template <>
struct pv_search_result<false> {
  using type = score_type;
};

template <>
struct pv_search_result<true> {
  using type = std::tuple<score_type, chess::move>;
};

template <bool is_root>
using pv_search_result_t = typename pv_search_result<is_root>::type;

struct search_worker;

struct internal_state {
  search_stack stack{chess::position_history{}, chess::board::start_pos()};
  nnue::eval::scratchpad_type scratchpad{};
  sided_history_heuristic hh{};
  eval_cache cache{};
  multicut_cache mc_cache{};
  std::unordered_map<chess::move, size_t, chess::move_hash> node_distribution{};

  std::atomic_bool go{false};
  std::atomic_size_t nodes{};
  std::atomic_size_t tb_hits{};
  std::atomic<depth_type> depth{};

  std::atomic<score_type> score{};

  std::atomic<chess::move::data_type> best_move{};
  std::atomic<chess::move::data_type> ponder_move{};

  bool keep_going() const { return go.load(std::memory_order::memory_order_relaxed); }

  template <size_t N>
  inline bool one_of() const {
    static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");
    constexpr size_t bit_pattern = N - 1;
    return (nodes & bit_pattern) == bit_pattern;
  }

  void reset() {
    stack = search_stack{chess::position_history{}, chess::board::start_pos()};
    hh.clear();
    cache.clear();
    node_distribution.clear();

    go.store(false);
    nodes.store(0);
    tb_hits.store(0);
    depth.store(0);
    score.store(0);
    best_move.store(chess::move::null().data);
  }
};

struct external_state {
  const nnue::weights* weights;
  std::shared_ptr<transposition_table> tt;
  std::shared_ptr<search_constants> constants;
  std::function<void(const search_worker&)> on_iter;
  std::function<void(const search_worker&)> on_update;

  external_state(
      const nnue::weights* weights_,
      std::shared_ptr<transposition_table> tt_,
      std::shared_ptr<search_constants> constants_,
      std::function<void(const search_worker&)>& on_iter_,
      std::function<void(const search_worker&)> on_update_)
      : weights{weights_}, tt{tt_}, constants{constants_}, on_iter{on_iter_}, on_update{on_update_} {}
};

struct search_worker {
  external_state external;
  internal_state internal{};

  template <bool is_pv, bool use_tt = true>
  score_type q_search(
      const stack_view& ss,
      nnue::eval_node& eval_node,
      const chess::board& bd,
      score_type alpha,
      const score_type& beta,
      const depth_type& elevation) {
    // callback on entering search function
    const bool should_update = internal.keep_going() && internal.one_of<nodes_per_update>();
    if (should_update) { external.on_update(*this); }

    ++internal.nodes;
    const bool is_check = bd.is_check();

    if (ss.is_two_fold(bd.hash())) { return draw_score; }
    if (bd.is_trivially_drawn()) { return draw_score; }

    const std::optional<transposition_table_entry> maybe = external.tt->find(bd.hash());
    if (maybe.has_value()) {
      const transposition_table_entry entry = maybe.value();
      const bool is_cutoff = (entry.bound() == bound_type::lower && entry.score() >= beta) || (entry.bound() == bound_type::exact) ||
                             (entry.bound() == bound_type::upper && entry.score() <= alpha);
      if (use_tt && is_cutoff) { return entry.score(); }
    }

    const auto [static_value, value] = [&] {
      const auto maybe_eval = internal.cache.find(bd.hash());
      const score_type static_value = is_check ? ss.loss_score() :
                                      !is_pv && maybe_eval.has_value() ?
                                                 maybe_eval.value() :
                                                 eval_node.evaluator().evaluate(bd.turn(), bd.phase<nnue::weights::parameter_type>());

      if (!is_check) { internal.cache.insert(bd.hash(), static_value); }

      score_type value = static_value;
      if (use_tt && maybe.has_value()) {
        if (maybe->bound() == bound_type::upper && static_value > maybe->score()) { value = maybe->score(); }
        if (maybe->bound() == bound_type::lower && static_value < maybe->score()) { value = maybe->score(); }
      }

      return std::tuple(static_value, value);
    }();

    if (!is_check && value >= beta) { return value; }
    if (ss.reached_max_height()) { return value; }

    move_orderer<chess::generation_mode::noisy_and_check> orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn())));
    if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

    alpha = std::max(alpha, value);
    score_type best_score = value;
    chess::move best_move = chess::move::null();

    ss.set_hash(bd.hash()).set_eval(static_value);
    int legal_count{0};
    for (const auto& [idx, mv] : orderer) {
      assert((mv != chess::move::null()));

      ++legal_count;
      if (!internal.keep_going()) { break; }

      if (!is_check && !bd.see_ge(mv, 0)) { continue; }

      const bool delta_prune = !is_pv && !is_check && !bd.see_gt(mv, 0) && ((value + external.constants->delta_margin()) < alpha);
      if (delta_prune) { continue; }

      const bool good_capture_prune = !is_pv && !is_check && !maybe.has_value() &&
                                      bd.see_ge(mv, external.constants->good_capture_prune_see_margin()) &&
                                      value + external.constants->good_capture_prune_score_margin() > beta;
      if (good_capture_prune) { return beta; }

      ss.set_played(mv);

      const chess::board bd_ = bd.forward(mv);
      external.tt->prefetch(bd_.hash());
      internal.cache.prefetch(bd_.hash());
      nnue::eval_node eval_node_ = eval_node.dirty_child(&bd, mv);

      const score_type score = -q_search<is_pv, use_tt>(ss.next(), eval_node_, bd_, -beta, -alpha, elevation + 1);

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          if (score < beta) { alpha = score; }
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }

      if (best_score >= beta) { break; }
    }

    if (legal_count == 0 && is_check) { return ss.loss_score(); }
    if (legal_count == 0) { return value; }

    if (use_tt && internal.keep_going()) {
      const bound_type bound = best_score >= beta ? bound_type::lower : bound_type::upper;
      const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, 0);
      external.tt->insert(entry);
    }

    return best_score;
  }

  template <bool is_pv, bool is_root = false>
  auto pv_search(
      const stack_view& ss,
      nnue::eval_node& eval_node,
      const chess::board& bd,
      score_type alpha,
      const score_type& beta,
      depth_type depth,
      const chess::player_type& reducer) -> pv_search_result_t<is_root> {
    auto make_result = [](const score_type& score, const chess::move& mv) {
      if constexpr (is_root) { return pv_search_result_t<is_root>{score, mv}; }
      if constexpr (!is_root) { return score; }
    };

    static_assert(!is_root || is_pv);
    assert(depth >= 0);

    // callback on entering search function
    const bool should_update = internal.keep_going() && (is_root || internal.one_of<nodes_per_update>());
    if (should_update) { external.on_update(*this); }

    // step 1. drop into qsearch if depth reaches zero
    if (depth <= 0) { return make_result(q_search<is_pv>(ss, eval_node, bd, alpha, beta, 0), chess::move::null()); }
    ++internal.nodes;

    // step 2. check if node is terminal
    const bool is_check = bd.is_check();

    if (!is_root && ss.is_two_fold(bd.hash())) { return make_result(draw_score, chess::move::null()); }
    if (!is_root && bd.is_trivially_drawn()) { return make_result(draw_score, chess::move::null()); }
    if (!is_root && bd.is_rule50_draw() && (!is_check || bd.generate_moves<chess::generation_mode::all>().size() != 0)) {
      return make_result(draw_score, chess::move::null());
    }

    if constexpr (is_root) {
      if (const syzygy::tb_dtz_result result = syzygy::probe_dtz(bd); result.success) { return make_result(result.score, result.move); }
    }

    const std::optional<multicut_info> mc_info = internal.mc_cache.find(bd.hash());
    const bool mc_cache_prune = !is_pv && !ss.has_excluded() && mc_info.has_value() && mc_info->depth >= depth + 1 && mc_info->score >= beta;
    if (mc_cache_prune) { return make_result(beta, chess::move::null()); }

    const score_type original_alpha = alpha;

    const std::optional<transposition_table_entry> maybe = !ss.has_excluded() ? external.tt->find(bd.hash()) : std::nullopt;
    if (maybe.has_value()) {
      const transposition_table_entry entry = maybe.value();
      const bool is_cutoff = !is_pv && entry.depth() >= depth &&
                             ((entry.bound() == bound_type::lower && entry.score() >= beta) || entry.bound() == bound_type::exact ||
                              (entry.bound() == bound_type::upper && entry.score() <= alpha));
      if (is_cutoff) { return make_result(entry.score(), entry.best_move()); }
    }

    if (const syzygy::tb_wdl_result result = syzygy::probe_wdl(bd); !is_root && result.success) {
      ++internal.tb_hits;

      switch (result.wdl) {
        case syzygy::wdl_type::loss: return make_result(ss.loss_score(), chess::move::null());
        case syzygy::wdl_type::draw: return make_result(draw_score, chess::move::null());
        case syzygy::wdl_type::win: return make_result(ss.win_score(), chess::move::null());
      }
    }

    // step 3. internal iterative reductions
    const bool should_iir = !maybe.has_value() && !ss.has_excluded() && depth >= external.constants->iir_depth();
    if (should_iir) { --depth; }

    // step 4. compute static eval and adjust appropriately if there's a tt hit
    const auto [static_value, value] = [&] {
      const auto maybe_eval = internal.cache.find(bd.hash());
      const score_type static_value = is_check ? ss.loss_score() :
                                      !is_pv && maybe_eval.has_value() ?
                                                 maybe_eval.value() :
                                                 eval_node.evaluator().evaluate(bd.turn(), bd.phase<nnue::weights::parameter_type>());

      if (!is_check) { internal.cache.insert(bd.hash(), static_value); }

      score_type value = static_value;
      if (maybe.has_value()) {
        if (maybe->bound() == bound_type::upper && static_value > maybe->score()) { value = maybe->score(); }
        if (maybe->bound() == bound_type::lower && static_value < maybe->score()) { value = maybe->score(); }
      }

      return std::tuple(static_value, value);
    }();

    // step 5. return static eval if max depth was reached
    if (ss.reached_max_height()) { return make_result(value, chess::move::null()); }

    // step 6. add position and static eval to stack
    ss.set_hash(bd.hash()).set_eval(static_value);
    const bool improving = !is_check && ss.improving();
    const chess::square_set threatened = bd.them_threat_mask();

    // step 7. static null move pruning
    const bool snm_prune = !is_pv && !ss.has_excluded() && !is_check && depth <= external.constants->snmp_depth() &&
                           value > beta + external.constants->snmp_margin(improving, threatened.any(), depth) && value > ss.loss_score();

    if (snm_prune) { return make_result(value, chess::move::null()); }

    // step 8. null move pruning
    const bool try_nmp =
        !is_pv && !ss.has_excluded() && !is_check && depth >= external.constants->nmp_depth() && value > beta && ss.nmp_valid() &&
        bd.has_non_pawn_material() && (!threatened.any() || depth >= 4) &&
        (!maybe.has_value() || (maybe->bound() == bound_type::lower && bd.is_legal<chess::generation_mode::all>(maybe->best_move()) &&
                                !bd.see_gt(maybe->best_move(), external.constants->nmp_see_threshold())));

    if (try_nmp) {
      ss.set_played(chess::move::null());
      const depth_type adjusted_depth = std::max(0, depth - external.constants->nmp_reduction(depth, beta, value));
      const score_type nmp_score =
          -pv_search<false>(ss.next(), eval_node, bd.forward(chess::move::null()), -beta, -beta + 1, adjusted_depth, chess::player_from(!bd.turn()));
      if (nmp_score >= beta) { return make_result(nmp_score, chess::move::null()); }
    }

    // step 9. initialize move orderer (setting tt move first if applicable)
    const chess::move killer = ss.killer();
    const chess::move follow = ss.follow();
    const chess::move counter = ss.counter();

    move_orderer<chess::generation_mode::all> orderer(
        move_orderer_data(&bd, &internal.hh.us(bd.turn())).set_killer(killer).set_follow(follow).set_counter(counter).set_threatened(threatened));

    if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

    // list of attempted moves for updating histories
    chess::move_list moves_tried{};

    // move loop
    score_type best_score = ss.loss_score();
    chess::move best_move = chess::move::null();

    bool did_double_extend{false};
    int legal_count{0};

    for (const auto& [idx, mv] : orderer) {
      assert((mv != chess::move::null()));

      ++legal_count;
      if (!internal.keep_going()) { break; }
      if (mv == ss.excluded()) { continue; }

      const size_t nodes_before = internal.nodes.load(std::memory_order_relaxed);
      ss.set_played(mv);

      const counter_type history_value = internal.hh.us(bd.turn()).compute_value(history::context{follow, counter, threatened}, mv);

      const chess::board bd_ = bd.forward(mv);

      const bool try_pruning = !is_root && idx >= 2 && best_score > max_mate_score;

      // step 10. pruning
      if (try_pruning) {
        const bool lm_prune = !bd_.is_check() && depth <= external.constants->lmp_depth() && idx > external.constants->lmp_count(improving, depth);

        if (lm_prune) { break; }

        const bool futility_prune =
            mv.is_quiet() && depth <= external.constants->futility_prune_depth() && value + external.constants->futility_margin(depth) < alpha;

        if (futility_prune) { continue; }

        const bool quiet_see_prune = mv.is_quiet() && depth <= external.constants->quiet_see_prune_depth() &&
                                     !bd.see_ge(mv, external.constants->quiet_see_prune_threshold(depth));

        if (quiet_see_prune) { continue; }

        const bool noisy_see_prune = mv.is_noisy() && depth <= external.constants->noisy_see_prune_depth() &&
                                     !bd.see_ge(mv, external.constants->noisy_see_prune_threshold(depth));

        if (noisy_see_prune) { continue; }

        const bool history_prune = mv.is_quiet() && history_value <= external.constants->history_prune_threshold(depth);

        if (history_prune) { continue; }
      }

      external.tt->prefetch(bd_.hash());
      internal.cache.prefetch(bd_.hash());
      internal.mc_cache.prefetch(bd_.hash());
      nnue::eval_node eval_node_ = eval_node.dirty_child(&bd, mv);

      // step 11. extensions
      bool multicut = false;
      const depth_type extension = [&, mv = mv] {
        const bool try_singular = !is_root && !ss.has_excluded() && depth >= external.constants->singular_extension_depth() && maybe.has_value() &&
                                  mv == maybe->best_move() && maybe->bound() != bound_type::upper &&
                                  maybe->depth() + external.constants->singular_extension_depth_margin() >= depth;

        if (try_singular) {
          const depth_type singular_depth = external.constants->singular_search_depth(depth);
          const score_type singular_beta = external.constants->singular_beta(maybe->score(), depth);
          ss.set_excluded(mv);
          const score_type excluded_score = pv_search<false>(ss, eval_node, bd, singular_beta - 1, singular_beta, singular_depth, reducer);
          ss.set_excluded(chess::move::null());

          if (!is_pv && excluded_score + external.constants->singular_double_extension_margin() < singular_beta) {
            did_double_extend = true;
            return 2;
          }
          if (excluded_score < singular_beta) { return 1; }

          if (excluded_score >= beta) {
            internal.mc_cache.insert(bd.hash(), multicut_info{excluded_score, depth});
            multicut = true;
          }
        }

        return 0;
      }();

      if (!is_root && multicut) { return make_result(beta, chess::move::null()); }

      const score_type score = [&, this, idx = idx, mv = mv] {
        const depth_type next_depth = depth + extension - 1;

        auto full_width = [&] { return -pv_search<is_pv>(ss.next(), eval_node_, bd_, -beta, -alpha, next_depth, reducer); };

        auto zero_width = [&](const depth_type& zw_depth) {
          const chess::player_type next_reducer = (is_pv || zw_depth < next_depth) ? chess::player_from(bd.turn()) : reducer;
          return -pv_search<false>(ss.next(), eval_node_, bd_, -alpha - 1, -alpha, zw_depth, next_reducer);
        };

        if (is_pv && idx == 0) { return full_width(); }

        depth_type lmr_depth;
        score_type zw_score;

        // step 12. late move reductions
        const bool try_lmr = !is_check && (mv.is_quiet() || !bd.see_ge(mv, 0)) && idx >= 2 && (depth >= external.constants->reduce_depth());
        if (try_lmr) {
          depth_type reduction = external.constants->reduction(depth, idx);

          // adjust reduction
          if (improving) { --reduction; }
          if (bd_.is_check()) { --reduction; }
          if (bd.is_passed_push(mv)) { --reduction; }
          if (bd.creates_threat(mv)) { --reduction; }

          if (!is_pv) { ++reduction; }
          if (did_double_extend) { ++reduction; }
          if (!bd.see_ge(mv, 0) && mv.is_quiet()) { ++reduction; }

          // if our opponent is the reducing player, an errant fail low will, at worst, induce a re-search
          // this idea is at least similar (maybe equivalent) to the "cutnode idea" found in Stockfish.
          if (is_player(reducer, !bd.turn())) { ++reduction; }

          if (mv.is_quiet()) { reduction += external.constants->history_reduction(history_value); }

          reduction = std::max(0, reduction);

          lmr_depth = std::max(1, next_depth - reduction);
          zw_score = zero_width(lmr_depth);
        }

        // search again at full depth if necessary
        if (!try_lmr || (zw_score > alpha && lmr_depth < next_depth)) { zw_score = zero_width(next_depth); }

        // search again with full window on pv nodes
        return (is_pv && (alpha < zw_score && zw_score < beta)) ? full_width() : zw_score;
      }();

      if (score < beta && (mv.is_quiet() || !bd.see_gt(mv, 0))) { moves_tried.push(mv); }

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          if (score < beta) { alpha = score; }
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }

      if constexpr (is_root) { internal.node_distribution[mv] += (internal.nodes.load(std::memory_order_relaxed) - nodes_before); }

      if (best_score >= beta) { break; }
    }

    if (legal_count == 0 && is_check) { return make_result(ss.loss_score(), chess::move::null()); }
    if (legal_count == 0) { return make_result(draw_score, chess::move::null()); }

    // step 13. update histories if appropriate and maybe insert a new transposition_table_entry
    if (internal.keep_going() && !ss.has_excluded()) {
      const bound_type bound = [&] {
        if (best_score >= beta) { return bound_type::lower; }
        if (is_pv && best_score > original_alpha) { return bound_type::exact; }
        return bound_type::upper;
      }();

      if (bound == bound_type::lower && (best_move.is_quiet() || !bd.see_gt(best_move, 0))) {
        internal.hh.us(bd.turn()).update(history::context{follow, counter, threatened}, best_move, moves_tried, depth);
        ss.set_killer(best_move);
      }

      const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, depth);
      external.tt->insert(entry);
    }

    return make_result(best_score, best_move);
  }

  void iterative_deepening_loop() {
    nnue::eval_node root_node = nnue::eval_node::clean_node([this] {
      nnue::eval result(external.weights, &internal.scratchpad, 0);
      internal.stack.root_pos().feature_full_refresh(result);
      return result;
    }());

    score_type alpha = -big_number;
    score_type beta = big_number;
    for (; internal.keep_going(); ++internal.depth) {
      internal.depth = std::min(max_depth, internal.depth.load());
      // update aspiration window once reasonable evaluation is obtained
      if (internal.depth >= external.constants->aspiration_depth()) {
        const score_type previous_score = internal.score;
        alpha = previous_score - aspiration_delta;
        beta = previous_score + aspiration_delta;
      }

      score_type delta = aspiration_delta;
      depth_type failed_high_count{0};

      for (;;) {
        internal.stack.clear_future();

        const depth_type adjusted_depth = std::max(1, internal.depth - failed_high_count);
        const auto [search_score, search_move] = pv_search<true, true>(
            stack_view::root(internal.stack), root_node, internal.stack.root_pos(), alpha, beta, adjusted_depth, chess::player_type::none);

        if (!internal.keep_going()) { break; }

        // update aspiration window if failing low or high
        if (search_score <= alpha) {
          beta = (alpha + beta) / 2;
          alpha = search_score - delta;
          failed_high_count = 0;
        } else if (search_score >= beta) {
          beta = search_score + delta;
          ++failed_high_count;
        } else {
          // store updated information
          internal.score.store(search_score);
          if (!search_move.is_null()) {
            internal.best_move.store(search_move.data);
            internal.ponder_move.store(internal.stack.ponder_move().data);
          }
          break;
        }

        // exponentially grow window
        delta += delta / 3;
      }

      // callback on iteration completion
      if (internal.keep_going()) { external.on_iter(*this); }
    }
  }

  size_t best_move_percent() const {
    constexpr size_t one_hundred = 100;
    const auto iter = internal.node_distribution.find(chess::move{internal.best_move});
    return iter != internal.node_distribution.end() ? (one_hundred * iter->second / internal.nodes.load()) : one_hundred;
  }

  size_t nodes() const { return internal.nodes.load(); }
  size_t tb_hits() const { return internal.tb_hits.load(); }
  depth_type depth() const { return internal.depth.load(); }
  chess::move best_move() const { return chess::move{internal.best_move.load()}; }
  chess::move ponder_move() const { return chess::move{internal.ponder_move.load()}; }

  score_type score() const { return internal.score.load(); }

  void go(const chess::position_history& hist, const chess::board& bd, const depth_type& start_depth) {
    internal.go.store(true);
    internal.node_distribution.clear();
    internal.nodes.store(0);
    internal.tb_hits.store(0);
    internal.depth.store(start_depth);
    internal.best_move.store(bd.generate_moves<>().begin()->data);
    internal.ponder_move.store(chess::move::null().data);
    internal.stack = search_stack(hist, bd);
  }

  void stop() { internal.go.store(false); }

  search_worker(
      const nnue::weights* weights,
      std::shared_ptr<transposition_table> tt,
      std::shared_ptr<search_constants> constants,
      std::function<void(const search_worker&)> on_iter = [](auto&&...) {},
      std::function<void(const search_worker&)> on_update = [](auto&&...) {})
      : external(weights, tt, constants, on_iter, on_update) {}
};

}  // namespace search