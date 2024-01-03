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

#include <array>
#include <cstdint>
#include <limits>

namespace zobrist {

using hash_type = std::uint64_t;
using half_hash_type = std::uint32_t;
using quarter_hash_type = std::uint16_t;

constexpr hash_type entropy_0 = 0x8c57d3cb77fabf02;
constexpr hash_type entropy_1 = 0xfe2951fb31cae837;
constexpr hash_type entropy_2 = 0x7b4f806efae54dc5;
constexpr hash_type entropy_3 = 0x2db772e1b89c6650;
constexpr hash_type entropy_4 = 0x19057b41fcb768a4;
constexpr hash_type entropy_5 = 0x1df555934cfcb8f5;

constexpr half_hash_type lower_half(const hash_type& hash) { return hash & std::numeric_limits<half_hash_type>::max(); }
constexpr half_hash_type upper_half(const hash_type& hash) { return (hash >> 32) & std::numeric_limits<half_hash_type>::max(); }
constexpr quarter_hash_type lower_quarter(const hash_type& hash) { return hash & std::numeric_limits<quarter_hash_type>::max(); }

struct xorshift_generator {
  hash_type seed_;

  [[nodiscard]] constexpr hash_type next() noexcept {
    seed_ ^= seed_ >> 12;
    seed_ ^= seed_ << 25;
    seed_ ^= seed_ >> 27;
    return seed_ * static_cast<hash_type>(2685821657736338717ull);
  }

  explicit constexpr xorshift_generator(const hash_type& seed) noexcept : seed_{seed} {}
};

}  // namespace zobrist
