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
#include <eval_cache.h>
#include <history_heuristic.h>
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
#include <mutex>
#include <thread>
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

struct internal_state {
  search_stack stack{chess::position_history{}, chess::board::start_pos()};
  sided_history_heuristic hh{};
  eval_cache cache{};
  std::unordered_map<chess::move, size_t, chess::move_hash> node_distribution{};

  std::atomic_size_t nodes{};
  std::atomic_size_t tb_hits{};
  std::atomic<depth_type> depth{};

  std::atomic<score_type> score{};

  std::atomic<chess::move::data_type> best_move{};
  std::atomic<chess::move::data_type> ponder_move{};

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
    nodes.store(0);
    tb_hits.store(0);
    depth.store(0);
    score.store(0);
    best_move.store(chess::move::null().data);
  }
};

template <typename T>
struct external_state {
  const nnue::weights* weights;
  std::shared_ptr<transposition_table> tt;
  std::shared_ptr<search_constants> constants;
  std::function<void(const T&)> on_iter;
  std::function<void(const T&)> on_update;

  external_state(
      const nnue::weights* weights_,
      std::shared_ptr<transposition_table> tt_,
      std::shared_ptr<search_constants> constants_,
      std::function<void(const T&)>& on_iter_,
      std::function<void(const T&)> on_update_)
      : weights{weights_}, tt{tt_}, constants{constants_}, on_iter{on_iter_}, on_update{on_update_} {}
};

template <bool is_active>
struct controlled_loop {
  std::atomic_bool go_{false};
  std::atomic_bool kill_{false};

  std::mutex control_mutex_{};
  std::condition_variable cv_{};
  std::thread active_;

  bool keep_going() const { return go_.load(std::memory_order_relaxed) && !kill_.load(std::memory_order_relaxed); }

  void loop_(std::function<void()> f) {
    for (;;) {
      if (kill_.load(std::memory_order_relaxed)) { break; }
      std::unique_lock<std::mutex> control_lk(control_mutex_);
      cv_.wait(control_lk, [this] { return go_.load(std::memory_order_relaxed) || kill_.load(std::memory_order_relaxed); });
      f();
      go_.store(false, std::memory_order_relaxed);
    }
  }

  void complete_iter() { go_.store(false, std::memory_order_relaxed); }

  void next(std::function<void()> on_next = [] {}) {
    {
      std::lock_guard<std::mutex> lk(control_mutex_);
      on_next();
      go_.store(true, std::memory_order_relaxed);
    }
    cv_.notify_one();
  }

  controlled_loop<is_active>& operator=(const controlled_loop<is_active>& other) = delete;
  controlled_loop<is_active>& operator=(controlled_loop<is_active>&& other) = delete;
  controlled_loop(const controlled_loop<is_active>& other) = delete;
  controlled_loop(controlled_loop<is_active>&& other) = delete;

  controlled_loop(std::function<void()> f)
      : active_([f, this] {
          (void)this;
          if constexpr (is_active) { loop_(f); }
        }) {}

  ~controlled_loop() {
    kill_.store(true, std::memory_order_relaxed);
    {
      std::lock_guard<std::mutex> lk(control_mutex_);
      kill_.store(true, std::memory_order_relaxed);
    }
    cv_.notify_one();
    active_.join();
  }
};

template <bool is_active = true>
struct search_worker {
  external_state<search_worker<is_active>> external;
  internal_state internal{};
  controlled_loop<is_active> loop;

  template <bool is_pv, bool use_tt = true>
  score_type q_search(
      const stack_view& ss, const nnue::eval& eval, const chess::board& bd, score_type alpha, const score_type& beta, const depth_type& elevation) {
    // callback on entering search function
    const bool should_update = loop.keep_going() && internal.one_of<nodes_per_update>();
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
      const score_type static_value = is_check                         ? ss.loss_score() :
                                      !is_pv && maybe_eval.has_value() ? maybe_eval.value() :
                                                                         eval.evaluate(bd.turn(), bd.phase<nnue::weights::parameter_type>());

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

    const chess::move_list list = bd.generate_moves<chess::generation_mode::noisy_and_check>();
    if (list.size() == 0 && is_check) { return ss.loss_score(); }
    if (list.size() == 0) { return value; }

    move_orderer orderer(move_orderer_data(&bd, &list, &internal.hh.us(bd.turn())));
    if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

    alpha = std::max(alpha, value);
    score_type best_score = value;
    chess::move best_move = chess::move::null();

    ss.set_hash(bd.hash()).set_eval(static_value);
    for (const auto& [idx, mv] : orderer) {
      assert((mv != chess::move::null()));
      if (!loop.keep_going()) { break; }

      const see_type see_value = bd.see<see_type>(mv);

      if (!is_check && see_value < 0) { continue; }

      const bool delta_prune = !is_pv && !is_check && (see_value <= 0) && ((value + external.constants->delta_margin()) < alpha);
      if (delta_prune) { continue; }

      const bool good_capture_prune = !is_pv && !is_check && !maybe.has_value() && see_value >= external.constants->good_capture_prune_see_margin() &&
                                      value + external.constants->good_capture_prune_score_margin() > beta;
      if (good_capture_prune) { return beta; }

      ss.set_played(mv);

      const chess::board bd_ = bd.forward(mv);
      external.tt->prefetch(bd_.hash());
      internal.cache.prefetch(bd_.hash());

      const nnue::eval eval_ = bd.apply_update(mv, eval);

      const score_type score = -q_search<is_pv, use_tt>(ss.next(), eval_, bd_, -beta, -alpha, elevation + 1);

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

    if (use_tt && loop.keep_going()) {
      const bound_type bound = best_score >= beta ? bound_type::lower : bound_type::upper;
      const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, 0);
      external.tt->insert(entry);
    }

    return best_score;
  }

  template <bool is_pv, bool is_root = false>
  auto pv_search(
      const stack_view& ss,
      const nnue::eval& eval,
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
    const bool should_update = loop.keep_going() && (is_root || internal.one_of<nodes_per_update>());
    if (should_update) { external.on_update(*this); }

    // step 1. drop into qsearch if depth reaches zero
    if (depth <= 0) { return make_result(q_search<is_pv>(ss, eval, bd, alpha, beta, 0), chess::move::null()); }
    ++internal.nodes;

    // step 2. check if node is terminal
    const bool is_check = bd.is_check();

    if (!is_root && ss.is_two_fold(bd.hash())) { return make_result(draw_score, chess::move::null()); }
    if (!is_root && bd.is_trivially_drawn()) { return make_result(draw_score, chess::move::null()); }
    if (!is_root && !is_check && bd.is_rule50_draw()) { return make_result(draw_score, chess::move::null()); }

    if constexpr (is_root) {
      if (const syzygy::tb_dtz_result result = syzygy::probe_dtz(bd); result.success) { return make_result(result.score, result.move); }
    }

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
      const score_type static_value = is_check                         ? ss.loss_score() :
                                      !is_pv && maybe_eval.has_value() ? maybe_eval.value() :
                                                                         eval.evaluate(bd.turn(), bd.phase<nnue::weights::parameter_type>());

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

    // step 8. prob pruning
    const bool prob_prune = !is_pv && !ss.has_excluded() && maybe.has_value() && depth >= external.constants->prob_prune_depth() &&
                            maybe->best_move().is_capture() && maybe->bound() == bound_type::lower &&
                            maybe->score() > beta + external.constants->prob_prune_margin() &&
                            maybe->depth() + external.constants->prob_prune_depth_margin(improving) >= depth;

    if (prob_prune) { return make_result(beta, chess::move::null()); }

    // step 9. null move pruning
    const bool try_nmp = !is_pv && !ss.has_excluded() && !is_check && depth >= external.constants->nmp_depth() && value > beta && ss.nmp_valid() &&
                         bd.has_non_pawn_material() && (!threatened.any() || depth >= 4) &&
                         (!maybe.has_value() || (maybe->bound() == bound_type::lower && bd.is_legal(maybe->best_move()) &&
                                                 bd.see<see_type>(maybe->best_move()) <= external.constants->nmp_see_threshold()));

    if (try_nmp) {
      ss.set_played(chess::move::null());
      const depth_type adjusted_depth = std::max(0, depth - external.constants->nmp_reduction(depth, beta, value));
      const score_type nmp_score =
          -pv_search<false>(ss.next(), eval, bd.forward(chess::move::null()), -beta, -beta + 1, adjusted_depth, chess::player_from(!bd.turn()));
      if (nmp_score >= beta) { return make_result(nmp_score, chess::move::null()); }
    }

    const chess::move_list list = bd.generate_moves<chess::generation_mode::all>();
    if (list.size() == 0 && is_check) { return make_result(ss.loss_score(), chess::move::null()); }
    if (list.size() == 0) { return make_result(draw_score, chess::move::null()); }

    // step 10. initialize move orderer (setting tt move first if applicable)
    const chess::move killer = ss.killer();
    const chess::move follow = ss.follow();
    const chess::move counter = ss.counter();

    move_orderer orderer(move_orderer_data(&bd, &list, &internal.hh.us(bd.turn()))
                             .set_killer(killer)
                             .set_follow(follow)
                             .set_counter(counter)
                             .set_threatened(threatened));

    if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

    // list of attempted moves for updating histories
    chess::move_list moves_tried{};

    // move loop
    score_type best_score = ss.loss_score();
    chess::move best_move = *list.begin();

    bool did_double_extend = false;

    for (const auto& [idx, mv] : orderer) {
      assert((mv != chess::move::null()));
      if (!loop.keep_going()) { break; }
      if (mv == ss.excluded()) { continue; }

      const size_t nodes_before = internal.nodes.load(std::memory_order_relaxed);
      ss.set_played(mv);

      const counter_type history_value = internal.hh.us(bd.turn()).compute_value(history::context{follow, counter, threatened}, mv);
      const see_type see_value = bd.see<see_type>(mv);

      const chess::board bd_ = bd.forward(mv);

      const bool try_pruning = !is_root && idx >= 2 && best_score > max_mate_score;

      // step 11. pruning
      if (try_pruning) {
        const bool lm_prune = !bd_.is_check() && depth <= external.constants->lmp_depth() && idx > external.constants->lmp_count(improving, depth);

        if (lm_prune) { break; }

        const bool futility_prune =
            mv.is_quiet() && depth <= external.constants->futility_prune_depth() && value + external.constants->futility_margin(depth) < alpha;

        if (futility_prune) { continue; }

        const bool quiet_see_prune =
            mv.is_quiet() && depth <= external.constants->quiet_see_prune_depth() && see_value < external.constants->quiet_see_prune_threshold(depth);

        if (quiet_see_prune) { continue; }

        const bool noisy_see_prune =
            mv.is_noisy() && depth <= external.constants->noisy_see_prune_depth() && see_value < external.constants->noisy_see_prune_threshold(depth);

        if (noisy_see_prune) { continue; }

        const bool history_prune = mv.is_quiet() && history_value <= external.constants->history_prune_threshold(depth);

        if (history_prune) { continue; }
      }

      external.tt->prefetch(bd_.hash());
      internal.cache.prefetch(bd_.hash());
      const nnue::eval eval_ = bd.apply_update(mv, eval);

      // step 12. extensions
      bool multicut = false;
      const depth_type extension = [&, mv = mv] {
        const bool try_singular = !is_root && !ss.has_excluded() && depth >= external.constants->singular_extension_depth() && maybe.has_value() &&
                                  mv == maybe->best_move() && maybe->bound() != bound_type::upper &&
                                  maybe->depth() + external.constants->singular_extension_depth_margin() >= depth;

        if (try_singular) {
          const depth_type singular_depth = external.constants->singular_search_depth(depth);
          const score_type singular_beta = external.constants->singular_beta(maybe->score(), depth);
          ss.set_excluded(mv);
          const score_type excluded_score = pv_search<false>(ss, eval, bd, singular_beta - 1, singular_beta, singular_depth, reducer);
          ss.set_excluded(chess::move::null());

          if (!is_pv && excluded_score + external.constants->singular_double_extension_margin() < singular_beta) {
            did_double_extend = true;
            return 2;
          }
          if (excluded_score < singular_beta) { return 1; }

          if (excluded_score >= beta) { multicut = true; }
        }

        return 0;
      }();

      if (!is_root && multicut) { return make_result(beta, chess::move::null()); }

      const score_type score = [&, this, idx = idx, mv = mv] {
        const depth_type next_depth = depth + extension - 1;

        auto full_width = [&] { return -pv_search<is_pv>(ss.next(), eval_, bd_, -beta, -alpha, next_depth, reducer); };

        auto zero_width = [&](const depth_type& zw_depth) {
          const chess::player_type next_reducer = (is_pv || zw_depth < next_depth) ? chess::player_from(bd.turn()) : reducer;
          return -pv_search<false>(ss.next(), eval_, bd_, -alpha - 1, -alpha, zw_depth, next_reducer);
        };

        if (is_pv && idx == 0) { return full_width(); }

        depth_type lmr_depth;
        score_type zw_score;

        // step 13. late move reductions
        const bool try_lmr = !is_check && (mv.is_quiet() || see_value < 0) && idx >= 2 && (depth >= external.constants->reduce_depth());
        if (try_lmr) {
          depth_type reduction = external.constants->reduction(depth, idx);

          // adjust reduction
          if (bd_.is_check()) { --reduction; }
          if (bd.is_passed_push(mv)) { --reduction; }
          if (improving) { --reduction; }
          if (!is_pv) { ++reduction; }
          if (did_double_extend) { ++reduction; }
          if (see_value < 0 && mv.is_quiet()) { ++reduction; }

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

      if (score < beta && (mv.is_quiet() || see_value <= 0)) { moves_tried.add_(mv); }

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

    // step 14. update histories if appropriate and maybe insert a new transposition_table_entry
    if (loop.keep_going() && !ss.has_excluded()) {
      const bound_type bound = [&] {
        if (best_score >= beta) { return bound_type::lower; }
        if (is_pv && best_score > original_alpha) { return bound_type::exact; }
        return bound_type::upper;
      }();

      if (bound == bound_type::lower && (best_move.is_quiet() || bd.see<see_type>(best_move) <= 0)) {
        internal.hh.us(bd.turn()).update(history::context{follow, counter, threatened}, best_move, moves_tried, depth);
        ss.set_killer(best_move);
      }

      const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, depth);
      external.tt->insert(entry);
    }

    return make_result(best_score, best_move);
  }

  void iterative_deepening_loop_() {
    const auto evaluator = [this] {
      nnue::eval result(external.weights);
      internal.stack.root_pos().feature_full_refresh(result);
      return result;
    }();

    score_type alpha = -big_number;
    score_type beta = big_number;
    for (; loop.keep_going(); ++internal.depth) {
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
            stack_view::root(internal.stack), evaluator, internal.stack.root_pos(), alpha, beta, adjusted_depth, chess::player_type::none);

        if (!loop.keep_going()) { break; }

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
          internal.best_move.store(search_move.data);
          internal.ponder_move.store(internal.stack.ponder_move().data);
          break;
        }

        // exponentially grow window
        delta += delta / 3;
      }

      // callback on iteration completion
      if (loop.keep_going()) { external.on_iter(*this); }
    }
  }

  size_t best_move_percent_() const { return 100 * internal.node_distribution.at(chess::move{internal.best_move}) / internal.nodes.load(); }
  size_t nodes() const { return internal.nodes.load(); }
  size_t tb_hits() const { return internal.tb_hits.load(); }
  depth_type depth() const { return internal.depth.load(); }
  chess::move best_move() const { return chess::move{internal.best_move.load()}; }
  chess::move ponder_move() const { return chess::move{internal.ponder_move.load()}; }

  score_type score() const { return internal.score.load(); }

  void go(const chess::position_history& hist, const chess::board& bd, const depth_type& start_depth) {
    loop.next([hist, bd, start_depth, this] {
      internal.node_distribution.clear();
      internal.nodes.store(0);
      internal.tb_hits.store(0);
      internal.depth.store(start_depth);
      internal.best_move.store(bd.generate_moves<>().begin()->data);
      internal.ponder_move.store(chess::move::null().data);
      internal.stack = search_stack(hist, bd);
    });
  }

  void stop() { loop.complete_iter(); }

  search_worker(
      const nnue::weights* weights,
      std::shared_ptr<transposition_table> tt,
      std::shared_ptr<search_constants> constants,
      std::function<void(const search_worker<is_active>&)> on_iter = [](auto&&...) {},
      std::function<void(const search_worker<is_active>&)> on_update = [](auto&&...) {})
      : external(weights, tt, constants, on_iter, on_update), loop([this] { iterative_deepening_loop_(); }) {}
};

struct worker_pool {
  static constexpr size_t primary_id = 0;

  const nnue::weights* weights_;
  std::shared_ptr<transposition_table> tt_{nullptr};
  std::shared_ptr<search_constants> constants_{nullptr};

  std::vector<std::unique_ptr<search_worker<>>> pool_{};

  void reset() {
    tt_->clear();
    for (auto& worker : pool_) { worker->internal.reset(); };
  }

  void resize(const size_t& new_size) {
    constants_->update_(new_size);
    const size_t old_size = pool_.size();
    pool_.resize(new_size);
    for (size_t i(old_size); i < new_size; ++i) { pool_[i] = std::make_unique<search_worker<>>(weights_, tt_, constants_); }
  }

  void go(const chess::position_history& hist, const chess::board& bd) {
    // increment table generation at start of search
    tt_->update_gen();
    for (size_t i(0); i < pool_.size(); ++i) {
      const depth_type start_depth = 1 + static_cast<depth_type>(i % 2);
      pool_[i]->go(hist, bd, start_depth);
    }
  }

  void stop() {
    for (auto& worker : pool_) { worker->stop(); }
  }

  size_t nodes() const {
    return std::accumulate(
        pool_.begin(), pool_.end(), static_cast<size_t>(0), [](const size_t& count, const auto& worker) { return count + worker->nodes(); });
  }

  size_t tb_hits() const {
    return std::accumulate(
        pool_.begin(), pool_.end(), static_cast<size_t>(0), [](const size_t& count, const auto& worker) { return count + worker->tb_hits(); });
  }

  search_worker<>& primary_worker() { return *pool_[primary_id]; }

  worker_pool(
      const nnue::weights* weights,
      size_t hash_table_size,
      std::function<void(const search_worker<>&)> on_iter = [](auto&&...) {},
      std::function<void(const search_worker<>&)> on_update = [](auto&&...) {})
      : weights_{weights} {
    tt_ = std::make_shared<transposition_table>(hash_table_size);
    constants_ = std::make_shared<search_constants>();
    pool_.push_back(std::make_unique<search_worker<>>(weights, tt_, constants_, on_iter, on_update));
  }
};

}  // namespace search
