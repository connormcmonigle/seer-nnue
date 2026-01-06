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
  static constexpr size_t N = 4096;
  static constexpr size_t mask = N - 1;
  static_assert((N & mask) == 0);
  static constexpr score_type eval_correction_scale = 256;

  std::array<score_type, N> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const zobrist::quarter_hash_type& feature_hash) noexcept { return feature_hash & mask; }

  [[nodiscard]] constexpr score_type correction_for(const zobrist::quarter_hash_type& feature_hash) const noexcept {
    const score_type raw_correction = data[hash_function(feature_hash)];
    return raw_correction / eval_correction_scale;
  }

  constexpr void update(const zobrist::quarter_hash_type& feature_hash, const score_type& error, const score_type& alpha) noexcept {
    constexpr score_type score_correction_limit = 65536;
    constexpr score_type filter_divisor = 256;

    const score_type filter_alpha = alpha;
    const score_type filter_c_alpha = filter_divisor - alpha;

    auto& correction = data[hash_function(feature_hash)];

    const score_type scaled_error = error * eval_correction_scale;
    correction = (correction * filter_c_alpha + scaled_error * filter_alpha) / filter_divisor;
    correction = std::clamp(correction, -score_correction_limit, score_correction_limit);
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
  static constexpr depth_type lookup_table_size = 32;

  static constexpr std::array<score_type, lookup_table_size> nonpv_alpha_lookup_table = [] {
    std::array<score_type, lookup_table_size> result{};
    for (depth_type depth{1}; depth < lookup_table_size; ++depth) {
      const double alpha_value = 1.0 - 1.0 / (1.0 + static_cast<double>(depth) / 8.0);
      result[depth] = static_cast<depth_type>(16.0 * alpha_value);
    }

    return result;
  }();

  static constexpr std::array<score_type, lookup_table_size> pv_alpha_lookup_table = [] {
    std::array<score_type, lookup_table_size> result{};
    for (depth_type depth{1}; depth < lookup_table_size; ++depth) {
      const double alpha_value = 1.0 - 1.0 / (1.0 + static_cast<double>(depth) / 4.0);
      result[depth] = static_cast<depth_type>(16.0 * alpha_value);
    }

    return result;
  }();

  std::array<eval_correction_history, N> histories_{};

  [[nodiscard]] constexpr score_type correction_for(const composite_feature_hash<N>& composite_hash) const noexcept {
    score_type result{};

    for (std::size_t i(0); i < N; ++i) {
      const zobrist::quarter_hash_type hash = composite_hash.hash(i);
      result += histories_[i].correction_for(hash);
    }

    return result;
  }

  template <bool is_pv>
  constexpr void update(const composite_feature_hash<N>& composite_hash, const bound_type& bound, const score_type& error, const depth_type& depth) noexcept {
    if (bound == bound_type::upper && error >= 0) { return; }
    if (bound == bound_type::lower && error <= 0) { return; }

    constexpr depth_type last_idx = lookup_table_size - 1;
    constexpr auto& alpha_lookup_table = is_pv ? pv_alpha_lookup_table : nonpv_alpha_lookup_table;

    const score_type alpha = alpha_lookup_table[std::min(last_idx, depth)];

    for (std::size_t i(0); i < N; ++i) {
      const zobrist::quarter_hash_type hash = composite_hash.hash(i);
      histories_[i].update(hash, error, alpha);
    }
  }

  void clear() noexcept {
    for (auto& history : histories_) { history.clear(); }
  }
};

constexpr std::size_t eval_correction_history_num_hashes = 3;
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
