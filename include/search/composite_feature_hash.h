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

#include <array>
#include <cstddef>

namespace search {

template <std::size_t N>
struct composite_feature_hash {
  std::array<zobrist::quarter_hash_type, N> hashes_;

  [[nodiscard]] constexpr zobrist::quarter_hash_type hash(const std::size_t& i) const noexcept { return hashes_[i]; }

  [[nodiscard]] constexpr zobrist::quarter_hash_type reduced() const noexcept {
    zobrist::quarter_hash_type value{};
    for (const zobrist::quarter_hash_type& hash : hashes_) { value ^= hash; }
    return value;
  }
};

template <typename... Ts>
[[nodiscard]] constexpr composite_feature_hash<sizeof...(Ts)> composite_feature_hash_of(const Ts&... ts) noexcept {
  return composite_feature_hash<sizeof...(Ts)>{{ts...}};
}

}  // namespace search
