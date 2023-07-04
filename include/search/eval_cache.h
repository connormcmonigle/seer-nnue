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

#include <search/search_constants.h>
#include <zobrist/util.h>

#include <array>
#include <optional>

namespace search {

struct eval_cache_entry {
  zobrist::half_hash_type hash{};
  score_type eval{};
};

struct eval_cache {
  static constexpr std::size_t size_mb = 8;
  static constexpr std::size_t N = (size_mb << 20) / sizeof(eval_cache_entry);
  static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");

  std::array<eval_cache_entry, N> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const zobrist::hash_type& hash) noexcept { return hash & (N - 1); }
  inline void prefetch(const zobrist::hash_type& hash) const noexcept { __builtin_prefetch(data.data() + hash_function(hash)); }

  [[nodiscard]] constexpr std::optional<score_type> find(const zobrist::hash_type& hash) const noexcept {
    if (data[hash_function(hash)].hash == zobrist::upper_half(hash)) { return data[hash_function(hash)].eval; }
    return std::nullopt;
  }

  void insert(const zobrist::hash_type& hash, const score_type& eval) noexcept {
    data[hash_function(hash)] = eval_cache_entry{zobrist::upper_half(hash), eval};
  }

  void clear() noexcept { return data.fill(eval_cache_entry{}); }
};

}  // namespace search