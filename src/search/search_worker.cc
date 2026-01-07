/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

#include <chess/move_zobrist_hasher.h>
#include <search/move_orderer.h>
#include <search/search_worker.h>
#include <search/syzygy.h>

namespace search {

template <bool is_pv, bool use_tt>
inline evaluate_info search_worker::evaluate(
    const stack_view& ss,
    nnue::eval_node& eval_node,
    const chess::board& bd,
    const std::optional<transposition_table_entry>& maybe) noexcept {
  const bool is_check = bd.is_check();

  const eval_cache_entry entry = [&] {
    constexpr zobrist::hash_type default_hash = zobrist::hash_type{};

    if (is_check) { return eval_cache_entry::make(default_hash, default_hash, ss.loss_score()); }
    if (const auto maybe_eval = internal.cache.find(bd.hash()); !is_pv && maybe_eval.has_value()) { return maybe_eval.value(); }

    const nnue::eval& evaluator = eval_node.evaluator();
    const zobrist::hash_type hash = bd.hash();

    const auto [eval_feature_hash, eval] = evaluator.evaluate(bd.turn(), bd.phase<nnue::weights::parameter_type>(), [](const auto& final_output) {
      constexpr std::size_t dimension = nnue::eval::final_output_type::dimension;
      return zobrist::zobrist_hasher<zobrist::quarter_hash_type, dimension>.compute_hash(
          [&final_output](const std::size_t& i) { return final_output.data[i] > nnue::weights::parameter_type{}; });
    });

    return eval_cache_entry::make(hash, eval_feature_hash, eval);
  }();

  const auto counter_move_hash = chess::counter_move_zobrist_hasher.compute_hash(ss.counter());
  const auto follow_move_hash = chess::follow_move_zobrist_hasher.compute_hash(ss.follow());

  const auto cont_feature_hash = zobrist::lower_quarter(counter_move_hash ^ follow_move_hash);
  const auto pawn_feature_hash = zobrist::lower_quarter(bd.pawn_hash());
  const auto eval_feature_hash = entry.eval_feature_hash();

  const auto feature_hash = composite_feature_hash_of(cont_feature_hash, pawn_feature_hash, eval_feature_hash);
  score_type static_value = entry.eval();

  if (!is_check) {
    internal.cache.insert(bd.hash(), entry);
    static_value += internal.correction.us(bd.turn()).correction_for(feature_hash);
  }

  score_type value = static_value;

  if (use_tt && maybe.has_value()) {
    if (maybe->bound() == bound_type::upper && static_value > maybe->score()) { value = maybe->score(); }
    if (maybe->bound() == bound_type::lower && static_value < maybe->score()) { value = maybe->score(); }
  }

  ss.set_eval(static_value);
  ss.set_eval_feature_hash(eval_feature_hash);
  return evaluate_info{feature_hash, static_value, value};
}

template <bool is_pv, bool use_tt>
score_type search_worker::q_search(
    const stack_view& ss,
    nnue::eval_node& eval_node,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    const depth_type& elevation) noexcept {
  // callback on entering search function
  const bool should_update = internal.keep_going() && internal.one_of<nodes_per_update>();
  if (should_update) { external.on_update(*this); }

  ++internal.nodes;
  const bool is_check = bd.is_check();

  if (bd.is_trivially_drawn()) { return draw_score; }
  if (ss.upcoming_cycle_exists(bd)) {
    if (draw_score >= beta) { return draw_score; }
    alpha = std::max(draw_score, alpha);
  }

  const std::optional<transposition_table_entry> maybe = external.tt->find(bd.hash());
  if (maybe.has_value()) {
    const transposition_table_entry entry = maybe.value();
    const bool is_cutoff = (entry.bound() == bound_type::lower && entry.score() >= beta) || (entry.bound() == bound_type::exact) ||
                           (entry.bound() == bound_type::upper && entry.score() <= alpha);
    if (use_tt && is_cutoff) { return entry.score(); }
  }

  const auto [feature_hash, static_value, value] = evaluate<is_pv, use_tt>(ss, eval_node, bd, maybe);

  if (!is_check && value >= beta) { return value; }
  if (ss.reached_max_height()) { return value; }

  move_orderer<chess::generation_mode::noisy_and_check> orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn())));
  if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

  alpha = std::max(alpha, value);
  score_type best_score = value;
  chess::move best_move = chess::move::null();

  ss.set_hash(bd.sided_hash());
  int legal_count{0};
  for (const auto& [idx, mv] : orderer) {
    ++legal_count;
    if (!internal.keep_going()) { break; }

    if (!is_check && !bd.see_ge(mv, 0)) { break; }

    const bool delta_prune = !is_pv && !is_check && !bd.see_gt(mv, 0) && ((value + external.constants->delta_margin()) < alpha);
    if (delta_prune) { break; }

    const bool good_capture_prune = !is_pv && !is_check && !maybe.has_value() && bd.see_ge(mv, external.constants->good_capture_prune_see_margin()) &&
                                    value + external.constants->good_capture_prune_score_margin() > beta;
    if (good_capture_prune) { return beta; }

    ss.set_played(mv);

    const chess::board bd_ = bd.forward(mv);
    external.tt->prefetch(bd_.hash());
    internal.cache.prefetch(bd_.hash());
    nnue::eval_node eval_node_ = eval_node.dirty_child(&internal.reset_cache, &bd, mv);

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

template <bool is_pv, bool is_root>
pv_search_result_t<is_root> search_worker::pv_search(
    const stack_view& ss,
    nnue::eval_node& eval_node,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    depth_type depth,
    const chess::player_type& reducer) noexcept {
  static_assert(!is_root || is_pv);

  auto make_result = [](const score_type& score, const chess::move& mv) {
    if constexpr (is_root) { return pv_search_result_t<is_root>{score, mv}; }
    if constexpr (!is_root) { return score; }
  };

  // callback on entering search function
  const bool should_update = internal.keep_going() && (is_root || internal.one_of<nodes_per_update>());
  if (should_update) { external.on_update(*this); }

  // step 1. drop into qsearch if depth reaches zero
  if (depth <= 0) { return make_result(q_search<is_pv>(ss, eval_node, bd, alpha, beta, 0), chess::move::null()); }
  ++internal.nodes;

  // step 2. check if node is terminal
  const bool is_check = bd.is_check();

  if (!is_root && bd.is_trivially_drawn()) { return make_result(draw_score, chess::move::null()); }
  if (!is_root && bd.is_rule50_draw() && (!is_check || bd.generate_moves<chess::generation_mode::all>().size() != 0)) {
    return make_result(draw_score, chess::move::null());
  }

  if (!is_root && ss.upcoming_cycle_exists(bd)) {
    if (draw_score >= beta) { return make_result(draw_score, chess::move::null()); }
    alpha = std::max(draw_score, alpha);
  }

  if constexpr (is_root) {
    if (const syzygy::tb_dtz_result result = syzygy::probe_dtz(bd); result.success) { return make_result(result.score, result.move); }
  }

  const std::optional<transposition_table_entry> maybe = !ss.has_excluded() ? external.tt->find(bd.hash()) : std::nullopt;
  if (maybe.has_value()) {
    const transposition_table_entry entry = maybe.value();
    const bool is_cutoff = !is_pv && entry.depth() >= depth &&
                           ((entry.bound() == bound_type::lower && entry.score() >= beta) || entry.bound() == bound_type::exact ||
                            (entry.bound() == bound_type::upper && entry.score() <= alpha));
    if (is_cutoff) { return make_result(entry.score(), entry.best_move()); }
  }

  const score_type original_alpha = alpha;
  const bool tt_pv = is_pv || (maybe.has_value() && maybe->tt_pv());

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
  const auto [feature_hash, static_value, value] = evaluate<is_pv>(ss, eval_node, bd, maybe);

  // step 5. return static eval if max depth was reached
  if (ss.reached_max_height()) { return make_result(value, chess::move::null()); }

  // step 6. add position to stack
  ss.set_hash(bd.sided_hash());
  const bool improving = !is_check && ss.improving();
  const chess::square_set threatened = bd.them_threat_mask();

  const bool try_razor = !is_pv && !is_check && !ss.has_excluded() && depth <= external.constants->razor_depth() &&
                         value + external.constants->razor_margin(depth) <= alpha;

  if (try_razor) {
    const score_type razor_score = q_search<false>(ss, eval_node, bd, alpha, alpha + 1, 0);
    if (razor_score <= alpha) { return make_result(razor_score, chess::move::null()); }
  }

  // step 7. static null move pruning
  const bool snm_prune = !is_pv && !ss.has_excluded() && !is_check && depth <= external.constants->snmp_depth() &&
                         value > beta + external.constants->snmp_margin(improving, threatened.any(), depth) && value > ss.loss_score();

  if (snm_prune) {
    const score_type adjusted_value = (beta + value) / 2;
    return make_result(adjusted_value, chess::move::null());
  }

  // step 8. null move pruning
  const bool try_nmp = !is_pv && !ss.has_excluded() && !is_check && depth >= external.constants->nmp_depth() && value > beta && ss.nmp_valid() &&
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

  // step 9. probcut pruning
  const depth_type probcut_depth = external.constants->probcut_search_depth(depth);
  const score_type probcut_beta = external.constants->probcut_beta(beta);
  const bool try_probcut = !is_pv && !ss.has_excluded() && depth >= external.constants->probcut_depth() &&
                           !(maybe.has_value() && maybe->best_move().is_quiet()) &&
                           !(maybe.has_value() && maybe->depth() >= probcut_depth && maybe->score() < probcut_beta);

  if (try_probcut) {
    move_orderer<chess::generation_mode::noisy_and_check> probcut_orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn())));
    if (maybe.has_value()) { probcut_orderer.set_first(maybe->best_move()); }

    for (const auto& [idx, mv] : probcut_orderer) {
      if (!internal.keep_going()) { break; }
      if (mv == ss.excluded()) { continue; }
      if (!bd.see_ge(mv, 0)) { continue; }

      ss.set_played(mv);

      const chess::board bd_ = bd.forward(mv);
      external.tt->prefetch(bd_.hash());
      internal.cache.prefetch(bd_.hash());
      nnue::eval_node eval_node_ = eval_node.dirty_child(&internal.reset_cache, &bd, mv);

      auto pv_score = [&] { return -pv_search<false>(ss.next(), eval_node_, bd_, -probcut_beta, -probcut_beta + 1, probcut_depth, reducer); };
      const score_type q_score = -q_search<false>(ss.next(), eval_node_, bd_, -probcut_beta, -probcut_beta + 1, 0);
      const score_type probcut_score = (q_score >= probcut_beta) ? pv_score() : q_score;

      if (probcut_score >= probcut_beta) { return make_result(probcut_score, mv); }
    }
  }

  // step 10. initialize move orderer (setting tt move first if applicable)
  const chess::move killer = ss.killer();
  const chess::move follow = ss.follow();
  const chess::move counter = ss.counter();
  const zobrist::hash_type pawn_hash = bd.pawn_hash();
  const zobrist::quarter_hash_type eval_feature_hash = ss.eval_feature_hash();

  move_orderer<chess::generation_mode::all> orderer(move_orderer_data(&bd, &internal.hh.us(bd.turn()))
                                                        .set_killer(killer)
                                                        .set_follow(follow)
                                                        .set_counter(counter)
                                                        .set_threatened(threatened)
                                                        .set_pawn_hash(pawn_hash)
                                                        .set_eval_feature_hash(eval_feature_hash));

  if (maybe.has_value()) { orderer.set_first(maybe->best_move()); }

  // list of attempted moves for updating histories
  chess::move_list moves_tried{};

  // move loop
  score_type best_score = ss.loss_score();
  chess::move best_move = chess::move::null();

  bool did_double_extend{false};
  int legal_count{0};

  for (const auto& [idx, mv] : orderer) {
    ++legal_count;
    if (!internal.keep_going()) { break; }
    if (mv == ss.excluded()) { continue; }

    const std::size_t nodes_before = internal.nodes.load(std::memory_order_relaxed);

    const history::context history_context = history::context{follow, counter, threatened, pawn_hash, eval_feature_hash};
    const counter_type history_value = internal.hh.us(bd.turn()).compute_value(history_context, mv);

    const chess::board bd_ = bd.forward(mv);

    const bool try_pruning = !is_root && idx >= 2 && best_score > max_mate_score;

    // step 11. pruning
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
    nnue::eval_node eval_node_ = eval_node.dirty_child(&internal.reset_cache, &bd, mv);

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
        const score_type excluded_score = pv_search<false>(ss, eval_node, bd, singular_beta - 1, singular_beta, singular_depth, reducer);
        ss.set_excluded(chess::move::null());

        if (!is_pv && excluded_score + external.constants->singular_double_extension_margin() < singular_beta) {
          did_double_extend = true;
          return 2;
        }

        if (excluded_score < singular_beta) { return 1; }
        if (excluded_score >= beta) { multicut = true; }
        if constexpr (!is_pv) { return -1; }
      }

      return 0;
    }();

    if (!is_root && multicut) { return make_result(beta, chess::move::null()); }

    ss.set_played(mv);

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

      // step 13. late move reductions
      const bool try_lmr = !is_check && (mv.is_quiet() || !bd.see_ge(mv, 0)) && idx >= 2 && (depth >= external.constants->reduce_depth());
      if (try_lmr) {
        depth_type reduction = external.constants->reduction(depth, idx);

        // adjust reduction
        if (improving) { --reduction; }
        if (bd_.is_check()) { --reduction; }
        if (bd.creates_threat(mv)) { --reduction; }
        if (mv == killer) { --reduction; }

        if (!tt_pv) { ++reduction; }
        if (did_double_extend) { ++reduction; }

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

  // step 14. update histories if appropriate and maybe insert a new transposition_table_entry
  if (internal.keep_going() && !ss.has_excluded()) {
    const bound_type bound = [&] {
      if (best_score >= beta) { return bound_type::lower; }
      if (is_pv && best_score > original_alpha) { return bound_type::exact; }
      return bound_type::upper;
    }();

    if (bound == bound_type::lower && (best_move.is_quiet() || !bd.see_gt(best_move, 0))) {
      internal.hh.us(bd.turn()).update(history::context{follow, counter, threatened, pawn_hash, eval_feature_hash}, best_move, moves_tried, depth);
      ss.set_killer(best_move);
    }

    if (!is_check && best_move.is_quiet()) {
      const score_type error = best_score - static_value;
      internal.correction.us(bd.turn()).update(feature_hash, bound, error, depth);
    }

    const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, depth, tt_pv);
    external.tt->insert(entry);
  }

  return make_result(best_score, best_move);
}

void search_worker::iterative_deepening_loop() noexcept {
  internal.reset_cache.reinitialize(external.weights);
  nnue::eval_node root_node = nnue::eval_node::clean_node([this] {
    nnue::eval result(external.weights, &internal.scratchpad, 0, 0);
    internal.stack.root().feature_full_reset(result);
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
    depth_type consecutive_failed_high_count{0};

    for (;;) {
      internal.stack.clear_future();

      const depth_type adjusted_depth = std::max(1, internal.depth - consecutive_failed_high_count);
      const auto [search_score, search_move] = pv_search<true, true>(
          stack_view::root(internal.stack), root_node, internal.stack.root(), alpha, beta, adjusted_depth, chess::player_type::none);

      if (!internal.keep_going()) { break; }

      // update aspiration window if failing low or high
      if (search_score <= alpha) {
        beta = (alpha + beta) / 2;
        alpha = search_score - delta;
        consecutive_failed_high_count = 0;
      } else if (search_score >= beta) {
        beta = search_score + delta;
        ++consecutive_failed_high_count;
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

}  // namespace search

template search::score_type search::search_worker::q_search<false, false>(
    const stack_view& ss,
    nnue::eval_node& eval_node,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    const depth_type& elevation);

template search::score_type search::search_worker::q_search<true, false>(
    const stack_view& ss,
    nnue::eval_node& eval_node,
    const chess::board& bd,
    score_type alpha,
    const score_type& beta,
    const depth_type& elevation);
