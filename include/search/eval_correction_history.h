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

#include <chess/types.h>
#include <search/search_constants.h>
#include <search/transposition_table.h>

#include <algorithm>

namespace search {

struct eval_correction_history_entry {
  zobrist::half_hash_type hash{};
  score_type correction{};

  static constexpr eval_correction_history_entry make(const zobrist::half_hash_type& hash) {
    return eval_correction_history_entry{hash, score_type{}};
  }
};

struct eval_correction_history {
  static constexpr size_t N = 131072;
  static constexpr size_t mask = N - 1;
  static_assert((N & mask) == 0);

  std::array<eval_correction_history_entry, N> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const zobrist::hash_type& feature_hash) noexcept { return feature_hash & mask; }
  inline void prefetch(const zobrist::hash_type& feature_hash) const noexcept { __builtin_prefetch(data.data() + hash_function(feature_hash)); }

  [[nodiscard]] constexpr score_type correction_for(const zobrist::hash_type& feature_hash) const noexcept {
    constexpr score_type correction_divisor = 8;

    if (data[hash_function(feature_hash)].hash == zobrist::upper_half(feature_hash)) {
      const score_type raw_correction = data[hash_function(feature_hash)].correction;
      return raw_correction / correction_divisor;
    }

    return score_type{};
  }

  void update(const zobrist::hash_type& feature_hash, const bound_type& bound, const score_type& delta) noexcept {
    static constexpr score_type delta_limit = 192;
    static constexpr score_type score_correction_limit = 256;
    static constexpr score_type ridge_regression_coefficient = 4;

    if (bound == bound_type::upper && delta <= 0) { return; }
    if (bound == bound_type::lower && delta >= 0) { return; }

    auto& entry = data[hash_function(feature_hash)];
    if (entry.hash != zobrist::upper_half(feature_hash)) { entry = eval_correction_history_entry::make(zobrist::upper_half(feature_hash)); }

    entry.correction -= std::clamp(delta, -delta_limit, delta_limit) + entry.correction / ridge_regression_coefficient;
    entry.correction = std::clamp(entry.correction, -score_correction_limit, score_correction_limit);
  }

  void clear() noexcept { return data.fill(eval_correction_history_entry{}); }
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