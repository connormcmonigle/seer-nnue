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

#include <chess/move.h>
#include <nnue/eval.h>
#include <nnue/feature_reset_cache.h>
#include <search/eval_cache.h>
#include <search/eval_correction_history.h>
#include <search/eval_delta_history.h>
#include <search/history_heuristic.h>
#include <search/search_stack.h>

#include <atomic>
#include <unordered_map>

namespace search {

struct search_worker_internal_state {
  nnue::sided_feature_reset_cache reset_cache{};
  search_stack stack{chess::board_history{}, chess::board::start_pos()};
  nnue::eval::scratchpad_type scratchpad{};
  sided_history_heuristic hh{};
  eval_cache cache{};
  sided_eval_delta_history delta{};
  sided_eval_correction_history correction{};
  std::unordered_map<chess::move, std::size_t, chess::move_hash> node_distribution{};

  std::atomic_bool go{false};
  std::atomic_size_t nodes{};
  std::atomic_size_t tb_hits{};
  std::atomic<depth_type> depth{};

  std::atomic<score_type> score{};

  std::atomic<chess::move::data_type> best_move{};
  std::atomic<chess::move::data_type> ponder_move{};

  [[nodiscard]] bool keep_going() const noexcept { return go.load(std::memory_order::memory_order_relaxed); }

  template <std::size_t N>
  [[nodiscard]] inline bool one_of() const noexcept {
    static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");
    constexpr std::size_t bit_pattern = N - 1;
    return (nodes & bit_pattern) == bit_pattern;
  }

  void reset() noexcept {
    stack = search_stack{chess::board_history{}, chess::board::start_pos()};
    hh.clear();
    cache.clear();
    correction.clear();
    node_distribution.clear();

    go.store(false);
    nodes.store(0);
    tb_hits.store(0);
    depth.store(0);
    score.store(0);
    best_move.store(chess::move::null().data);
  }
};

}  // namespace search
