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
#include <zobrist/util.h>

#include <algorithm>
#include <vector>

namespace chess {

struct sided_zobrist_hash : public sided<sided_zobrist_hash, zobrist::hash_type> {
  zobrist::hash_type white;
  zobrist::hash_type black;
  zobrist::hash_type hash;

  constexpr sided_zobrist_hash() : white{}, black{}, hash{} {}

  constexpr sided_zobrist_hash(const zobrist::hash_type& white, const zobrist::hash_type& black, const zobrist::hash_type& hash) noexcept
      : white{white}, black{black}, hash{hash} {}
};

template <typename T, std::size_t N>
struct circular_fixed_size_history {
  static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");

  using value_type = T;
  static constexpr std::size_t mask = N - 1;

  std::size_t size_{};
  T data_[N]{};

  [[nodiscard]] constexpr T& at(const std::size_t& idx) noexcept { return data_[idx & mask]; }
  [[nodiscard]] constexpr const T& at(const std::size_t& idx) const noexcept { return data_[idx & mask]; }

  [[nodiscard]] constexpr T& future_at(const std::size_t& height) noexcept { return data_[(size_ + height) & mask]; }
  [[nodiscard]] constexpr const T& future_at(const std::size_t& height) const noexcept { return data_[(size_ + height) & mask]; }

  [[nodiscard]] std::size_t size() const noexcept { return size_; }
  [[nodiscard]] std::size_t future_size(const std::size_t& height) const noexcept { return size_ + height; }

  [[maybe_unused]] constexpr circular_fixed_size_history<T, N>& clear() noexcept {
    size_ = std::size_t{};
    return *this;
  }

  [[maybe_unused]] constexpr circular_fixed_size_history<T, N>& push(const T& value) noexcept {
    data_[size_ & mask] = value;
    ++size_;
    return *this;
  }

  template <typename U>
  [[nodiscard]] constexpr std::size_t count(const std::size_t& height, const U& value) const noexcept {
    std::size_t count_value{};
#pragma omp simd reduction(+ : count_value)
    for (std::size_t i = 0; i < (size_ + height); ++i) { count_value += value == data_[i & mask].hash; }
    return count_value;
  }
};

using board_history = circular_fixed_size_history<sided_zobrist_hash, 4096>;

}  // namespace chess
