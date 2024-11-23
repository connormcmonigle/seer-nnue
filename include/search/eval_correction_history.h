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

struct eval_correction_history {
  static constexpr size_t N = 16384;
  static constexpr size_t mask = N - 1;
  static_assert((N & mask) == 0);

  std::array<score_type, N> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const zobrist::quarter_hash_type& feature_hash) noexcept { return feature_hash & mask; }

  [[nodiscard]] constexpr score_type raw_correction_for(const zobrist::quarter_hash_type& feature_hash) const noexcept {
    return data[hash_function(feature_hash)];
  }

  constexpr void update(const zobrist::quarter_hash_type& feature_hash, const score_type& error, const depth_type& depth) noexcept {
    constexpr score_type bonus_limit = 256;
    constexpr score_type bonus_divisor = 8;
    constexpr score_type update_divisor = 1024;

    const score_type bonus = std::clamp((error * depth) / bonus_divisor, -bonus_limit, bonus_limit);

    score_type& history = data[hash_function(feature_hash)];
    history += (bonus - history * std::abs(bonus) / update_divisor);
  }

  void clear() noexcept { return data.fill(score_type{}); }
};

template <std::size_t N>
struct composite_feature_hash {
  std::array<zobrist::quarter_hash_type, N> hashes_;

  [[nodiscard]] constexpr zobrist::quarter_hash_type hash(const std::size_t& i) const noexcept { return hashes_[i]; }
};

template <typename... Ts>
[[nodiscard]] constexpr composite_feature_hash<sizeof...(Ts)> composite_feature_hash_of(const Ts&... ts) noexcept {
  return composite_feature_hash<sizeof...(Ts)>{{ts...}};
}

template <std::size_t N>
struct composite_eval_correction_history {
  std::array<eval_correction_history, N> histories_{};

  [[nodiscard]] constexpr score_type correction_for(const composite_feature_hash<N>& composite_hash) const noexcept {
    constexpr score_type mysterious_constant = 80;
    constexpr score_type eval_correction_scale = 512;

    score_type result{};

    for (std::size_t i(0); i < N; ++i) {
      const zobrist::quarter_hash_type hash = composite_hash.hash(i);
      result += mysterious_constant * histories_[i].raw_correction_for(hash);
    }

    return result / eval_correction_scale;
  }

  constexpr void update(const composite_feature_hash<N>& composite_hash, const bound_type& bound, const score_type& error, const depth_type& depth) noexcept {
    if (bound == bound_type::upper && error >= 0) { return; }
    if (bound == bound_type::lower && error <= 0) { return; }

    for (std::size_t i(0); i < N; ++i) {
      const zobrist::quarter_hash_type hash = composite_hash.hash(i);
      histories_[i].update(hash, error, depth);
    }
  }

  void clear() noexcept {
    for (auto& history : histories_) { history.clear(); }
  }
};

constexpr std::size_t eval_correction_history_num_hashes = 2;
struct sided_eval_correction_history
    : public chess::sided<sided_eval_correction_history, composite_eval_correction_history<eval_correction_history_num_hashes>> {
  using hash_type = composite_feature_hash<eval_correction_history_num_hashes>;
  composite_eval_correction_history<eval_correction_history_num_hashes> white;
  composite_eval_correction_history<eval_correction_history_num_hashes> black;

  void clear() noexcept {
    white.clear();
    black.clear();
  }

  sided_eval_correction_history() : white{}, black{} {}
};

}  // namespace search
