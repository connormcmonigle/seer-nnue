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
#include <zobrist/util.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>

namespace search {
using probability_type = std::uint32_t;

struct filter_config {
  static constexpr probability_type filter_divisor = 256;

  probability_type alpha;
  probability_type c_alpha;

  template <probability_type alpha>
  [[nodiscard]] static constexpr filter_config make() {
    static_assert(alpha < filter_divisor);
    return filter_config{alpha, filter_divisor - alpha};
  }
};

constexpr filter_config success_filter_config = filter_config::make<16>();
constexpr filter_config failure_filter_config = filter_config::make<1>();

struct adaptive_reduction_history {
  static constexpr depth_type depth_limit = 128;
  static constexpr probability_type filter_divisor = 256;
  static constexpr probability_type filter_alpha = 1;
  static constexpr probability_type filter_c_alpha = 255;

  static constexpr probability_type probability_divisor = 1024;
  static constexpr probability_type probability_mask = probability_divisor - 1;
  static constexpr probability_type probability_epsilon = 16;

  static_assert((probability_divisor & probability_mask) == 0);

  zobrist::xorshift_generator xorshift_generator_;
  std::array<probability_type, depth_limit + 1> reduce_less_probabilities_;

  [[nodiscard]] constexpr bool should_reduce_less(const depth_type& depth) noexcept {
    const depth_type limited_depth = std::min(depth_limit, depth);
    const zobrist::half_hash_type uniform_random_number = zobrist::lower_half(xorshift_generator_.next());

    return (probability_mask & uniform_random_number) < reduce_less_probabilities_[limited_depth];
  }

  constexpr void update(const depth_type& depth, const bool success) noexcept {
    const depth_type limited_depth = std::min(depth_limit, depth);

    const filter_config config = success ? success_filter_config : failure_filter_config;
    const probability_type target_probability = success ? probability_divisor : 0;
    probability_type& probability = reduce_less_probabilities_[limited_depth];

    probability = (config.alpha * target_probability + config.c_alpha * probability) / filter_divisor;
    probability = std::max(probability_epsilon, probability);
  }

  void clear() noexcept {
    xorshift_generator_ = zobrist::xorshift_generator{zobrist::entropy_0};
    reduce_less_probabilities_.fill(probability_epsilon);
  }

  adaptive_reduction_history() noexcept : xorshift_generator_{zobrist::entropy_0}, reduce_less_probabilities_{} {
    reduce_less_probabilities_.fill(probability_epsilon);
  }
};

struct sided_adaptive_reduction_history : public chess::sided<sided_adaptive_reduction_history, adaptive_reduction_history> {
  adaptive_reduction_history white;
  adaptive_reduction_history black;

  void clear() noexcept {
    white.clear();
    black.clear();
  }

  sided_adaptive_reduction_history() noexcept : white{}, black{} {}
};

}  // namespace search