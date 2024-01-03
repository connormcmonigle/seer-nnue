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

#pragma once

#include <chess/board.h>
#include <chess/move.h>
#include <chess/move_list.h>
#include <nnue/eval_node.h>
#include <search/search_constants.h>
#include <search/search_stack.h>
#include <search/search_worker_external_state.h>
#include <search/search_worker_internal_state.h>

#include <functional>
#include <memory>

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

struct evaluation_info {
  zobrist::quarter_hash_type feature_hash;
  score_type static_value;
  score_type value;
};

struct search_worker {
  search_worker_external_state external;
  search_worker_internal_state internal{};

  template <bool is_pv, bool use_tt = true>
  [[nodiscard]] inline evaluation_info evaluate_(
      const stack_view& ss,
      nnue::eval_node& eval_node,
      const chess::board& bd,
      const std::optional<transposition_table_entry>& maybe) noexcept;

  template <bool is_pv, bool use_tt = true>
  [[nodiscard]] score_type q_search(
      const stack_view& ss,
      nnue::eval_node& eval_node,
      const chess::board& bd,
      score_type alpha,
      const score_type& beta,
      const depth_type& elevation) noexcept;

  template <bool is_pv, bool is_root = false>
  [[nodiscard]] pv_search_result_t<is_root> pv_search(
      const stack_view& ss,
      nnue::eval_node& eval_node,
      const chess::board& bd,
      score_type alpha,
      const score_type& beta,
      depth_type depth,
      const chess::player_type& reducer) noexcept;

  void iterative_deepening_loop() noexcept;

  [[nodiscard]] std::size_t best_move_percent() const noexcept {
    constexpr std::size_t one_hundred = 100;
    const auto iter = internal.node_distribution.find(chess::move{internal.best_move});
    return iter != internal.node_distribution.end() ? (one_hundred * iter->second / internal.nodes.load()) : one_hundred;
  }

  [[nodiscard]] std::size_t nodes() const noexcept { return internal.nodes.load(); }
  [[nodiscard]] std::size_t tb_hits() const noexcept { return internal.tb_hits.load(); }
  [[nodiscard]] depth_type depth() const noexcept { return internal.depth.load(); }
  [[nodiscard]] chess::move best_move() const noexcept { return chess::move{internal.best_move.load()}; }
  [[nodiscard]] chess::move ponder_move() const noexcept { return chess::move{internal.ponder_move.load()}; }
  [[nodiscard]] score_type score() const noexcept { return internal.score.load(); }

  void go(const chess::board_history& hist, const chess::board& bd, const depth_type& start_depth) noexcept {
    internal.go.store(true);
    internal.node_distribution.clear();
    internal.nodes.store(0);
    internal.tb_hits.store(0);
    internal.depth.store(start_depth);
    internal.best_move.store(bd.generate_moves<>().begin()->data);
    internal.ponder_move.store(chess::move::null().data);
    internal.stack = search_stack(hist, bd);
  }

  void stop() noexcept { internal.go.store(false); }

  search_worker(
      const nnue::quantized_weights* weights,
      std::shared_ptr<transposition_table> tt,
      std::shared_ptr<search_constants> constants,
      std::function<void(const search_worker&)> on_iter = [](auto&&...) {},
      std::function<void(const search_worker&)> on_update = [](auto&&...) {}) noexcept
      : external(weights, tt, constants, on_iter, on_update) {}
};

}  // namespace search