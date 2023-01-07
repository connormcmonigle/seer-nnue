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

namespace search {

template <typename T>
struct local_cache_entry {
  zobrist::half_hash_type hash{};
  T data{};
};

struct multicut_info {
  score_type score;
  depth_type depth;
};

template <typename T, size_t size_mb>
struct local_cache {
  static constexpr size_t N = (size_mb << 20) / sizeof(local_cache_entry<T>);

  std::array<local_cache_entry<T>, N> entries{};

  constexpr size_t hash_function(const zobrist::hash_type& hash) const { return hash % entries.size(); }

  void prefetch(const zobrist::hash_type& hash) const { __builtin_prefetch(entries.data() + hash_function(hash)); }

  std::optional<T> find(const zobrist::hash_type& hash) const {
    if (entries[hash_function(hash)].hash == zobrist::upper_half(hash)) { return entries[hash_function(hash)].data; }
    return std::nullopt;
  }

  void insert(const zobrist::hash_type& hash, const T& data) { entries[hash_function(hash)] = local_cache_entry<T>{zobrist::upper_half(hash), data}; }

  void clear() { return entries.fill(local_cache_entry<T>{}); }
};

using eval_cache = local_cache<score_type, 6>;
using multicut_cache = local_cache<multicut_info, 6>;

}  // namespace search