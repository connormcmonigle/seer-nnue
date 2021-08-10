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

#include <search_constants.h>
#include <zobrist_util.h>

#include <array>
#include <optional>

namespace chess {

struct eval_cache_entry {
  zobrist::half_hash_type hash{};
  search::score_type eval{};
};

struct eval_cache {
  static constexpr size_t size_mb = 6;
  static constexpr size_t N = (size_mb << 20) / sizeof(eval_cache_entry);

  std::array<eval_cache_entry, N> data{};

  std::optional<search::score_type> find(const zobrist::hash_type& hash) const {
    if (data[hash % N].hash == zobrist::upper_half(hash)) { return data[hash % N].eval; }
    return std::nullopt;
  }

  void insert(const zobrist::hash_type& hash, const search::score_type& eval) { data[hash % N] = eval_cache_entry{zobrist::upper_half(hash), eval}; }

  void clear() { return data.fill(eval_cache_entry{}); }
};

}  // namespace chess