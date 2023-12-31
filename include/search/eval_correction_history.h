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
#include <chess/types.h>
#include <search/search_constants.h>
#include <search/transposition_table.h>

#include <algorithm>

namespace search {

struct piece_eval_correction_history {
  static constexpr size_t N = 1024;
  static constexpr size_t mask = N - 1;
  static_assert((N & mask) == 0);
  static constexpr score_type eval_correction_scale = 256;

  std::array<score_type, N> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const zobrist::hash_type& piece_hash) noexcept { return piece_hash & mask; }

  [[nodiscard]] constexpr score_type correction_for(const zobrist::hash_type& piece_hash) const noexcept {
    const score_type raw_correction = data[hash_function(piece_hash)];
    return raw_correction / eval_correction_scale;
  }

  constexpr void update(const zobrist::hash_type& piece_hash, const score_type& error) noexcept {
    constexpr score_type score_correction_limit = 65536;

    constexpr score_type filter_alpha = 1;
    constexpr score_type filter_c_alpha = 255;
    constexpr score_type filter_divisor = filter_alpha + filter_c_alpha;

    auto& correction = data[hash_function(piece_hash)];

    const score_type scaled_error = error * eval_correction_scale;
    correction = (correction * filter_c_alpha + scaled_error * filter_alpha) / filter_divisor;
    correction = std::clamp(correction, -score_correction_limit, score_correction_limit);
  }

  void clear() noexcept { data.fill(score_type{}); }
};

struct eval_correction_history {
  piece_eval_correction_history pawn_{};
  piece_eval_correction_history knight_{};
  piece_eval_correction_history bishop_{};
  piece_eval_correction_history rook_{};
  piece_eval_correction_history queen_{};
  piece_eval_correction_history king_{};

  [[nodiscard]] constexpr score_type correction_for(const chess::board& bd) const noexcept {
    constexpr score_type correction_divisor = 6;

    score_type correction{};
    chess::over_types([&, this](const chess::piece_type& pt) {
      const zobrist::hash_type piece_hash = bd.piece_hash(pt);
      correction += chess::get_member(pt, *this).correction_for(piece_hash);
    });

    return correction / correction_divisor;
  }

  constexpr void update(const chess::board& bd, const bound_type& bound, const score_type& error) noexcept {
    if (bound == bound_type::upper && error >= 0) { return; }
    if (bound == bound_type::lower && error <= 0) { return; }

    chess::over_types([&, this](const chess::piece_type& pt) {
      const zobrist::hash_type piece_hash = bd.piece_hash(pt);
      chess::get_member(pt, *this).update(piece_hash, error);
    });
  }

  void clear() noexcept {
    chess::over_types([this](const chess::piece_type& pt) { chess::get_member(pt, *this).clear(); });
  }
};

struct sided_eval_correction_history : public chess::sided<sided_eval_correction_history, eval_correction_history> {
  eval_correction_history white;
  eval_correction_history black;

  void clear() noexcept {
    white.clear();
    black.clear();
  }

  sided_eval_correction_history() : white{}, black{} {}
};

}  // namespace search
