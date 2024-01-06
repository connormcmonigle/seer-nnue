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

#include <chess/square.h>
#include <chess/types.h>

#include <array>

namespace chess {

template <color C>
struct pawn_info {};

template <>
struct pawn_info<color::white> {
  static constexpr int start_rank_idx = 1;
  static constexpr int last_rank_idx = 7;
  static constexpr int double_rank_idx = 3;
  static constexpr square_set start_rank = generate_rank(start_rank_idx);
  static constexpr square_set last_rank = generate_rank(last_rank_idx);
  static constexpr square_set double_rank = generate_rank(double_rank_idx);
  static constexpr std::array<delta, 2> attack = {delta{-1, 1}, delta{1, 1}};
  static constexpr delta step = delta{0, 1};
};

template <>
struct pawn_info<color::black> {
  static constexpr int start_rank_idx = 6;
  static constexpr int last_rank_idx = 0;
  static constexpr int double_rank_idx = 4;
  static constexpr square_set start_rank = generate_rank(start_rank_idx);
  static constexpr square_set last_rank = generate_rank(last_rank_idx);
  static constexpr square_set double_rank = generate_rank(double_rank_idx);
  static constexpr std::array<delta, 2> attack = {delta{-1, -1}, delta{1, -1}};
  static constexpr delta step = delta{0, -1};
};

}  // namespace chess
