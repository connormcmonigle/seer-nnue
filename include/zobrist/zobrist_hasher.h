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

#include <zobrist/util.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <utility>

namespace zobrist {

namespace detail {

template <typename T, std::size_t N>
struct zobrist_hash_source_impl {
  std::array<T, N> data{};

  constexpr zobrist_hash_source_impl(zobrist::xorshift_generator generator) noexcept {
    for (auto& elem : data) { elem = generator.next(); }
  }
};

template <typename T, std::size_t N>
inline constexpr zobrist_hash_source_impl<T, N> zobrist_hash_source = zobrist_hash_source_impl<T, N>{xorshift_generator(zobrist::entropy_0)};

}  // namespace detail

template <typename T, std::size_t N>
struct zobrist_hasher_impl {
  static constexpr T initial_hash_value = T{};
  static constexpr std::size_t feature_cardinality = N;

  template <typename F>
  [[nodiscard]] constexpr T compute_hash(F&& indicator_function) const noexcept {
    T hash = initial_hash_value;

#pragma omp simd
    for (std::size_t i = 0; i < N; ++i) {
      const T mask = static_cast<T>(indicator_function(i));
      hash ^= mask * detail::zobrist_hash_source<T, N>.data[i];
    }

    return hash;
  }
};

template <typename T, std::size_t N>
inline constexpr zobrist_hasher_impl<T, N> zobrist_hasher = zobrist_hasher_impl<T, N>{};

}  // namespace zobrist
