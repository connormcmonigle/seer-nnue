/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

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

#include <cstdint>
#include <utility>

namespace bit {

template <typename T, size_t B0, size_t B1 = 8 * sizeof(T)>
struct range {
  static_assert(B0 < B1, "wrong bit order");
  using type = T;
  static constexpr size_t first = B0;
  static constexpr size_t last = B1;

  template <typename I>
  static constexpr T get(const I& i) {
    constexpr int num_bits = 8 * sizeof(I);
    static_assert(B1 <= num_bits, "integral type accessed by bit::range::get has insufficient bits");
    constexpr I one = static_cast<I>(1);
    constexpr I b0 = static_cast<I>(first);
    constexpr I b1 = static_cast<I>(last);
    constexpr I mask = (one << (b1 - b0)) - one;
    return static_cast<T>((i >> b0) & mask);
  }

  template <typename I>
  static constexpr void set(I& i, const T& info) {
    constexpr int num_bits = 8 * sizeof(I);
    static_assert(B1 <= num_bits, "integral type accessed by bit::range::set has insufficient bits");
    constexpr I one = static_cast<I>(1);
    constexpr I b0 = static_cast<I>(first);
    constexpr I b1 = static_cast<I>(last);
    const I info_ = static_cast<I>(info);
    constexpr I mask = ((one << (b1 - b0)) - one) << b0;
    i &= ~mask;
    i |= (info_ << b0) & mask;
  }
};

template <size_t B>
using flag = range<bool, B, B + 1>;

template <typename R>
using next_flag = flag<R::last>;

template <typename R, typename T, size_t width = 8 * sizeof(T)>
using next_range = range<T, R::last, R::last + width>;

}  // namespace bit
