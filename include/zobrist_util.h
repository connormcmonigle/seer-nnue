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

#include <array>
#include <limits>
#include <cstdint>
#include <random>

namespace zobrist{


using hash_type = std::uint64_t;
constexpr std::mt19937::result_type seed = 0x019ec6dc;
  
inline hash_type random_bit_string(){
  static std::mt19937 gen(seed);
  constexpr hash_type a = std::numeric_limits<hash_type>::min();
  constexpr hash_type b = std::numeric_limits<hash_type>::max();
  std::uniform_int_distribution<hash_type> dist(a, b);
  return dist(gen);
}

}
