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

#include <search/transposition_table.h>

namespace search {

void transposition_table::clear() noexcept {
  for (auto& elem : data) { elem = bucket_type{}; }
}

void transposition_table::resize(const std::size_t& size) noexcept {
  clear();
  data.resize(size * one_mb, bucket_type{});
}

void transposition_table::update_gen() noexcept {
  using gen_type = transposition_table_entry::gen_type;
  constexpr gen_type limit = gen_type{1} << transposition_table_entry::gen_bits;
  current_gen = (current_gen + 1) % limit;
}

// clang-format off

__attribute__((no_sanitize("thread")))
transposition_table& transposition_table::insert(const transposition_table_entry& entry) noexcept {
  constexpr depth_type offset = 2;
  const transposition_table_entry::gen_type gen = current_gen.load(std::memory_order_relaxed);

  transposition_table_entry* to_replace = data[hash_function(entry.key())].to_replace(gen, entry.key());

  const bool should_replace =
      (entry.bound() == bound_type::exact) || (entry.key() != to_replace->key()) || ((entry.depth() + offset) >= to_replace->depth());

  if (should_replace) { *to_replace = transposition_table_entry(entry).set_gen(gen).merge(*to_replace); }

  return *this;
}

// clang-format on

// clang-format off

__attribute__((no_sanitize("thread")))
std::optional<transposition_table_entry> transposition_table::find(const zobrist::hash_type& key) noexcept {
  const transposition_table_entry::gen_type gen = current_gen.load(std::memory_order_relaxed);
  return data[hash_function(key)].match(gen, key);
}

// clang-format on

}  // namespace search
