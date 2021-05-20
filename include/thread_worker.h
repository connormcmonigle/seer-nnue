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
#include <transposition_table.h>

#include <atomic>
#include <condition_variable>
#include <cstring>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <vector>

namespace chess {

template <bool is_root>
struct pv_search_result {};
template <>
struct pv_search_result<false> {
  using type = search::score_type;
};
template <>
struct pv_search_result<true> {
  using type = std::tuple<search::score_type, move>;
};

template <bool is_root>
using pv_search_result_t = typename pv_search_result<is_root>::type;

struct internal_state {
  search::stack stack{position_history{}, board::start_pos()};
  sided_history_heuristic hh{};
  eval_cache cache{};

  std::atomic_bool is_stable{false};
  std::atomic_size_t nodes{};
  std::atomic<search::depth_type> depth{};

  std::atomic<search::score_type> score{};
  std::atomic_uint32_t best_move{};

  inline bool one_of_512() const {
    constexpr size_t bit_pattern = 511;
    return (nodes & bit_pattern) == bit_pattern;
  }
};

template <typename T>
struct external_state {
  const nnue::weights<typename T::weight_type>* weights;
  std::shared_ptr<transposition_table> tt;
  std::shared_ptr<search::constants> constants;
  std::function<void(const T&)> on_iter;
  std::function<void(const T&)> on_update;

  external_state(
      const nnue::weights<typename T::weight_type>* weights_,
      std::shared_ptr<transposition_table> tt_,
      std::shared_ptr<search::constants> constants_,
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

template <typename T, bool is_active = true>
struct thread_worker {
  using weight_type = T;

  external_state<thread_worker<T, is_active>> external;
  internal_state internal{};
  controlled_loop<is_active> loop;

  template <bool is_pv>
  search::score_type q_search(
      const search::stack_view& ss,
      const nnue::eval<T>& eval,
      const board& bd,
      search::score_type alpha,
      const search::score_type& beta,
      const search::depth_type& elevation) {
    // callback on entering search function
    const bool should_update = loop.keep_going() && internal.one_of_512();
    if (should_update) { external.on_update(*this); }

    ++internal.nodes;
    const move_list list = bd.generate_loud_moves();
    const bool is_check = bd.is_check();

    if (list.size() == 0 && is_check) { return ss.effective_mate_score(); }
    if (ss.is_two_fold(bd.hash())) { return search::draw_score; }
    if (bd.is_trivially_drawn()) { return search::draw_score; }

    move_orderer orderer(move_orderer_data{move::null(), move::null(), move::null(), &bd, list, &internal.hh.us(bd.turn())});

    const std::optional<transposition_table_entry> maybe = external.tt->find(bd.hash());
    if (maybe.has_value()) {
      const transposition_table_entry entry = maybe.value();
      const bool is_cutoff =
          (entry.score() >= beta && entry.bound() == bound_type::lower) || (entry.score() <= alpha && entry.bound() == bound_type::upper);
      if (is_cutoff) { return entry.score(); }
      orderer.set_first(entry.best_move());
    }

    const search::score_type static_eval = [&] {
      const auto maybe_eval = internal.cache.find(bd.hash());
      const search::score_type val = is_check                         ? ss.effective_mate_score() :
                                     !is_pv && maybe_eval.has_value() ? maybe_eval.value() :
                                                                        eval.evaluate(bd.turn());

      if (!is_check) { internal.cache.insert(bd.hash(), val); }

      if (maybe.has_value()) {
        if (maybe->bound() == bound_type::upper && val > maybe->score()) { return maybe->score(); }
        if (maybe->bound() == bound_type::lower && val < maybe->score()) { return maybe->score(); }
      }
      return val;
    }();

    if (list.size() == 0 || static_eval >= beta) { return static_eval; }
    if (ss.reached_max_height()) { return static_eval; }

    alpha = std::max(alpha, static_eval);
    search::score_type best_score = static_eval;
    move best_move = move::null();

    ss.set_hash(bd.hash()).set_eval(static_eval);
    for (const auto& [idx, mv] : orderer) {
      assert((mv != move::null()));
      if (!loop.keep_going() || best_score >= beta) { break; }
      if (!is_check && bd.see<search::see_type>(mv) < 0) { continue; }
      ss.set_played(mv);

      const board bd_ = bd.forward(mv);
      external.tt->prefetch(bd_.hash());

      const nnue::eval<T> eval_ = bd.apply_update(mv, eval);

      const search::score_type score = -q_search<is_pv>(ss.next(), eval_, bd_, -beta, -alpha, elevation + 1);

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          if (score < beta) { alpha = score; }
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }
    }

    if (loop.keep_going()) {
      const bound_type bound = best_score >= beta ? bound_type::lower : bound_type::upper;
      const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, 0);
      external.tt->insert(entry);
    }

    return best_score;
  }

  template <bool is_pv, bool is_root = false>
  auto pv_search(
      const search::stack_view& ss,
      const nnue::eval<T>& eval,
      const board& bd,
      search::score_type alpha,
      const search::score_type& beta,
      search::depth_type depth) -> pv_search_result_t<is_root> {
    auto make_result = [](const search::score_type& score, const move& mv) {
      if constexpr (is_root) { return pv_search_result_t<is_root>{score, mv}; }
      if constexpr (!is_root) { return score; }
    };

    assert(depth >= 0);

    // callback on entering search function
    // callback on entering search function
    const bool should_update = loop.keep_going() && (is_root || internal.one_of_512());
    if (should_update) { external.on_update(*this); }

    // step 1. drop into qsearch if depth reaches zero
    if (depth <= 0) { return make_result(q_search<is_pv>(ss, eval, bd, alpha, beta, 0), move::null()); }
    ++internal.nodes;

    // step 2. check if node is terminal
    const move_list list = bd.generate_moves();
    const bool is_check = bd.is_check();
    if (list.size() == 0 && is_check) { return make_result(ss.effective_mate_score(), move::null()); }
    if (list.size() == 0) { return make_result(search::draw_score, move::null()); }
    if (!is_root && ss.is_two_fold(bd.hash())) { return make_result(search::draw_score, move::null()); }
    if (!is_root && bd.is_trivially_drawn()) { return make_result(search::draw_score, move::null()); }

    // step 3. initialize move orderer (setting tt move first if applicable)
    // and check for tt entry + tt induced cutoff on nonpv nodes
    const move killer = ss.killer();
    const move follow = ss.follow();
    const move counter = ss.counter();

    move_orderer orderer(move_orderer_data{killer, follow, counter, &bd, list, &internal.hh.us(bd.turn())});
    const std::optional<transposition_table_entry> maybe = external.tt->find(bd.hash());
    if (maybe.has_value()) {
      const transposition_table_entry entry = maybe.value();
      const bool is_cutoff =
          !is_pv && entry.depth() >= depth && (entry.score() >= beta ? (entry.bound() == bound_type::lower) : (entry.bound() == bound_type::upper));
      if (is_cutoff) { return make_result(entry.score(), entry.best_move()); }
      orderer.set_first(entry.best_move());
    }

    // step 4. compute static eval and adjust appropriately if there's a tt hit
    const search::score_type static_eval = [&] {
      const auto maybe_eval = internal.cache.find(bd.hash());
      const search::score_type val = is_check                         ? ss.effective_mate_score() :
                                     !is_pv && maybe_eval.has_value() ? maybe_eval.value() :
                                                                        eval.evaluate(bd.turn());

      if (!is_check) { internal.cache.insert(bd.hash(), val); }

      if (maybe.has_value()) {
        if (maybe->bound() == bound_type::upper && val > maybe->score()) { return maybe->score(); }
        if (maybe->bound() == bound_type::lower && val < maybe->score()) { return maybe->score(); }
      }
      return val;
    }();

    // step 5. return static eval if max depth was reached
    if (ss.reached_max_height()) { return make_result(static_eval, move::null()); }

    // step 6. add position and static eval to stack
    ss.set_hash(bd.hash()).set_eval(static_eval);
    const bool improving = ss.improving();

    // step 7. static null move pruning
    const bool snm_prune = !is_root && !is_pv && !is_check && depth <= external.constants->snmp_depth() &&
                           static_eval > beta + external.constants->snmp_margin(improving, depth) && static_eval > ss.effective_mate_score();

    if (snm_prune) { return make_result(static_eval, move::null()); }

    // step 8. null move pruning
    const bool try_nmp = !is_root && !is_pv && !is_check && depth >= external.constants->nmp_depth() && static_eval > beta && ss.nmp_valid() &&
                         bd.has_non_pawn_material();

    if (try_nmp) {
      ss.set_played(move::null());
      const search::depth_type R = external.constants->R(depth);
      const search::depth_type adjusted_depth = std::max(0, depth - R);
      const search::score_type nmp_score = -pv_search<is_pv>(ss.next(), eval, bd.forward(move::null()), -beta, -alpha, adjusted_depth);
      if (nmp_score > beta) { return make_result(nmp_score, move::null()); }
    }

    // list of attempted quiets for updating histories
    move_list quiets_tried{};

    // move loop
    search::score_type best_score = ss.effective_mate_score();
    move best_move = *list.begin();

    for (const auto& [idx, mv] : orderer) {
      assert((mv != move::null()));
      if (!loop.keep_going() || best_score >= beta) { break; }
      ss.set_played(mv);

      const search::counter_type history_value = internal.hh.us(bd.turn()).compute_value(follow, counter, mv);

      const board bd_ = bd.forward(mv);

      const bool try_pruning =
          !is_root && !is_pv && !bd_.is_check() && !is_check && idx != 0 && mv.is_quiet() && best_score > ss.effective_mate_score();

      // step 9. pruning
      if (try_pruning) {
        const bool lm_prune = depth <= external.constants->lmp_depth() && quiets_tried.size() > external.constants->lmp_count(improving, depth);

        if (lm_prune) { continue; }

        const bool history_prune =
            depth <= external.constants->history_prune_depth() && history_value <= external.constants->history_prune_threshold(improving, depth);

        if (history_prune) { continue; }

        const bool futility_prune =
            depth <= external.constants->futility_prune_depth() && static_eval + external.constants->futility_margin(depth) < alpha;

        if (futility_prune) { continue; }
      }

      external.tt->prefetch(bd_.hash());
      const nnue::eval<T> eval_ = bd.apply_update(mv, eval);

      // step 10. extensions
      const search::depth_type extension = [&, mv = mv] {
        const bool check_ext = bd.see<search::see_type>(mv) > 0 && bd_.is_check();

        if (check_ext) { return 1; }

        const bool history_ext = !is_root && maybe.has_value() && mv == maybe->best_move() && mv.is_quiet() &&
                                 depth >= external.constants->history_extension_depth() &&
                                 history_value >= external.constants->history_extension_threshold();

        if (history_ext) { return 1; }

        return 0;
      }();

      const search::score_type score = [&, this, idx = idx, mv = mv] {
        const search::depth_type next_depth = depth + extension - 1;
        auto full_width = [&] { return -pv_search<is_pv>(ss.next(), eval_, bd_, -beta, -alpha, next_depth); };

        const bool try_lmr =
            !is_check && (mv.is_quiet() || bd.see<search::see_type>(mv) < 0) && idx != 0 && (depth >= external.constants->reduce_depth());
        search::score_type zw_score{};

        // step 11. late move reductions
        if (try_lmr) {
          search::depth_type reduction = external.constants->reduction(depth, idx);

          // adjust reduction
          if (bd_.is_check()) { --reduction; }
          if (bd.is_passed_push(mv)) { --reduction; }
          if (!improving) { ++reduction; }
          if (!is_pv) { ++reduction; }
          if (bd.see<search::see_type>(mv) < 0 && mv.is_quiet()) { ++reduction; }

          if (mv.is_quiet()) { reduction += external.constants->history_reduction(history_value); }

          reduction = std::max(reduction, 0);

          const search::depth_type lmr_depth = std::max(1, next_depth - reduction);
          zw_score = -pv_search<false>(ss.next(), eval_, bd_, -alpha - 1, -alpha, lmr_depth);
        }

        // search again at full depth if necessary
        if (!try_lmr || (try_lmr && (zw_score > alpha))) { zw_score = -pv_search<false>(ss.next(), eval_, bd_, -alpha - 1, -alpha, next_depth); }

        // search again with full window on pv nodes
        const bool interior = zw_score > alpha && zw_score < beta;
        return (interior && is_pv) ? full_width() : zw_score;
      }();

      if (score < beta && mv.is_quiet()) { quiets_tried.add_(mv); }

      if (score > best_score) {
        best_score = score;
        best_move = mv;
        if (score > alpha) {
          if (score < beta) { alpha = score; }
          if constexpr (is_pv) { ss.prepend_to_pv(mv); }
        }
      }
    }

    // step 12. update histories if appropriate and maybe insert a new transposition_table_entry
    if (loop.keep_going()) {
      if (best_score >= beta) {
        const transposition_table_entry entry(bd.hash(), bound_type::lower, best_score, best_move, depth);
        external.tt->insert(entry);
        if (best_move.is_quiet()) {
          internal.hh.us(bd.turn()).update(follow, counter, best_move, quiets_tried, depth);
          ss.set_killer(best_move);
        }
      } else {
        const transposition_table_entry entry(bd.hash(), bound_type::upper, best_score, best_move, depth);
        external.tt->insert(entry);
      }
    }

    return make_result(best_score, best_move);
  }

  void iterative_deepening_loop_() {
    const auto evaluator = [this] {
      nnue::eval<T> result(external.weights);
      internal.stack.root_pos().show_init(result);
      return result;
    }();

    search::score_type alpha = -search::big_number;
    search::score_type beta = search::big_number;
    for (; loop.keep_going() && internal.depth <= (external.constants->max_depth()); ++internal.depth) {
      // update aspiration window once reasonable evaluation is obtained
      if (internal.depth >= external.constants->aspiration_depth()) {
        const search::score_type previous_score = internal.score;
        alpha = previous_score - search::aspiration_delta;
        beta = previous_score + search::aspiration_delta;
      }

      search::score_type delta = search::aspiration_delta;
      search::depth_type failed_high_count{0};

      for (;;) {
        internal.stack.clear_future();

        const search::depth_type adjusted_depth = std::max(1, internal.depth - failed_high_count);
        const auto [search_score, search_move] =
            pv_search<true, true>(search::stack_view::root(internal.stack), evaluator, internal.stack.root_pos(), alpha, beta, adjusted_depth);

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

          internal.is_stable.store(std::abs(score() - search_score) <= search::stability_threshold && internal.best_move.load() == search_move.data);

          internal.score.store(search_score);
          internal.best_move.store(search_move.data);
          break;
        }

        // exponentially grow window
        delta += delta / 3;
      }

      // callback on iteration completion
      if (loop.keep_going()) { external.on_iter(*this); }
    }
  }

  bool is_stable() const { return internal.is_stable.load(); }
  size_t nodes() const { return internal.nodes.load(); }
  search::depth_type depth() const { return internal.depth.load(); }
  move best_move() const { return move{internal.best_move.load()}; }
  search::score_type score() const { return internal.score.load(); }

  void go(const position_history& hist, const board& bd, const search::depth_type& start_depth) {
    loop.next([hist, bd, start_depth, this] {
      internal.nodes.store(0);
      internal.depth.store(start_depth);
      internal.is_stable.store(false);
      internal.best_move.store(bd.generate_moves().begin()->data);
      internal.stack = search::stack(hist, bd);
      internal.hh.clear();
      internal.cache.clear();
    });
  }

  void stop() { loop.complete_iter(); }

  thread_worker(
      const nnue::weights<T>* weights,
      std::shared_ptr<transposition_table> tt,
      std::shared_ptr<search::constants> constants,
      std::function<void(const thread_worker<T, is_active>&)> on_iter = [](auto&&...) {},
      std::function<void(const thread_worker<T, is_active>&)> on_update = [](auto&&...) {})
      : external(weights, tt, constants, on_iter, on_update), loop([this]() { iterative_deepening_loop_(); }) {}
};

template <typename T>
struct worker_pool {
  static constexpr size_t primary_id = 0;

  const nnue::weights<T>* weights_;
  std::shared_ptr<transposition_table> tt_{nullptr};
  std::shared_ptr<search::constants> constants_{nullptr};

  std::vector<std::unique_ptr<thread_worker<T>>> pool_{};

  void resize(const size_t& new_size) {
    constants_->update_(new_size);
    const size_t old_size = pool_.size();
    pool_.resize(new_size);
    for (size_t i(old_size); i < new_size; ++i) { pool_[i] = std::make_unique<thread_worker<T>>(weights_, tt_, constants_); }
  }

  void go(const position_history& hist, const board& bd) {
    // increment table generation at start of search
    tt_->update_gen();
    for (size_t i(0); i < pool_.size(); ++i) {
      const search::depth_type start_depth = 1 + static_cast<search::depth_type>(i % 2);
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

  thread_worker<T>& primary_worker() { return *pool_[primary_id]; }

  worker_pool(
      const nnue::weights<T>* weights,
      size_t hash_table_size,
      std::function<void(const thread_worker<T, true>&)> on_iter = [](auto&&...) {},
      std::function<void(const thread_worker<T, true>&)> on_update = [](auto&&...) {})
      : weights_{weights} {
    tt_ = std::make_shared<transposition_table>(hash_table_size);
    constants_ = std::make_shared<search::constants>();
    pool_.push_back(std::make_unique<thread_worker<T>>(weights, tt_, constants_, on_iter, on_update));
  }
};

}  // namespace chess
