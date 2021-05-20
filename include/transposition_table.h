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

#include <bit_range.h>
#include <move.h>
#include <search_constants.h>
#include <zobrist_util.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace chess {

enum class bound_type { upper, lower };

constexpr std::string_view bound_type_name(const bound_type& type) {
  switch (type) {
    case bound_type::upper: return "upper";
    case bound_type::lower: return "lower";
    default: return "?";
  }
}

struct transposition_table_entry {
  using generation_type = std::uint32_t;

  using type_ = bit::range<bound_type, 0, 2>;
  using score_ = bit::next_range<type_, search::score_type>;
  using best_move_ = bit::next_range<score_, move::data_type, move::width>;

  zobrist::hash_type key_;
  zobrist::hash_type value_;
  search::depth_type depth_;

  // table assigned field
  generation_type gen{0};

  const zobrist::hash_type& key() const { return key_; }
  const zobrist::hash_type& value() const { return value_; }
  search::depth_type depth() const { return depth_; }

  bound_type bound() const { return type_::get(value_); }

  search::score_type score() const { return score_::get(value_); }

  move best_move() const { return move{best_move_::get(value_)}; }

  transposition_table_entry(
      const zobrist::hash_type& key, const bound_type& type, const search::score_type& score, const chess::move& mv, const search::depth_type& depth)
      : key_{key}, value_{0}, depth_{depth} {
    type_::set(value_, type);
    score_::set(value_, score);
    best_move_::set(value_, mv.data);
  }

  transposition_table_entry(const zobrist::hash_type& k, const zobrist::hash_type& v, const search::depth_type& depth)
      : key_{k}, value_{v}, depth_{depth} {}
  transposition_table_entry() : key_{0}, value_{0}, depth_{0} {}
};

std::ostream& operator<<(std::ostream& ostr, const transposition_table_entry& entry) {
  ostr << "transposition_table_entry(key=" << entry.key();
  ostr << ", key^value=" << (entry.key() ^ entry.value());
  ostr << ", best_move=" << entry.best_move();
  ostr << ", bound=" << bound_type_name(entry.bound());
  ostr << ", score=" << entry.score();
  return ostr << ", depth=" << entry.depth() << ')';
}

struct transposition_table {
  using iterator = std::vector<transposition_table_entry>::const_iterator;

  static constexpr size_t bucket_size = 4;
  static constexpr size_t idx_mask = ~0x3;

  static constexpr size_t one_mb = (static_cast<size_t>(1) << static_cast<size_t>(20)) / sizeof(transposition_table_entry);
  std::vector<transposition_table_entry> data;
  std::atomic<transposition_table_entry::generation_type> current_gen{0};

  std::vector<transposition_table_entry>::const_iterator begin() const { return data.cbegin(); }
  std::vector<transposition_table_entry>::const_iterator end() const { return data.cend(); }

  void resize(size_t size) {
    const size_t new_size = size * one_mb - ((size * one_mb) % bucket_size);
    data.resize(new_size, transposition_table_entry{});
  }

  void clear() {
    std::transform(data.begin(), data.end(), data.begin(), [](auto) { return transposition_table_entry{}; });
  }

  void update_gen() {
    constexpr auto limit = std::numeric_limits<transposition_table_entry::generation_type>::max();
    current_gen = (current_gen + 1) % limit;
  }

  size_t hash_function(const zobrist::hash_type& hash) const { return idx_mask & (hash % data.size()); }

  __attribute__((no_sanitize("thread"))) size_t find_idx(const zobrist::hash_type& hash, const size_t& base_idx) const {
    for (size_t i{base_idx}; i < (base_idx + bucket_size); ++i) {
      if ((data[i].key() ^ data[i].value()) == hash) { return i; }
    }
    return base_idx;
  }

  __attribute__((no_sanitize("thread"))) size_t replacement_idx(const zobrist::hash_type& hash, const size_t& base_idx) {
    auto heuristic = [this](const size_t& idx) {
      constexpr int m0 = 1;
      constexpr int m1 = 512;
      return m0 * data[idx].depth() - m1 * static_cast<int>(current_gen.load() != data[idx].gen);
    };

    size_t worst_idx = base_idx;
    int worst_score = std::numeric_limits<int>::max();

    for (size_t i{base_idx}; i < base_idx + bucket_size; ++i) {
      if ((data[i].key() ^ data[i].value()) == hash) { return i; }
      const int score = heuristic(i);
      if (score < worst_score) {
        worst_idx = i;
        worst_score = score;
      }
    }
    return worst_idx;
  }

  __attribute__((no_sanitize("thread"))) transposition_table& insert(const transposition_table_entry& entry) {
    const size_t base_idx = hash_function(entry.key());
    const size_t idx = replacement_idx(entry.key(), base_idx);
    assert(idx < data.size());
    data[idx] = entry;
    data[idx].key_ ^= entry.value();
    data[idx].gen = current_gen.load();
    return *this;
  }

  __attribute__((no_sanitize("thread"))) std::optional<transposition_table_entry> find(const zobrist::hash_type& key) const {
    const size_t base_idx = hash_function(key);
    const size_t idx = find_idx(key, base_idx);
    assert(idx < data.size());
    const transposition_table_entry result = data[idx];
    return (key == (result.key() ^ result.value())) ? std::optional(result) : std::nullopt;
  }

  transposition_table(size_t size) : data(size * one_mb - ((size * one_mb) % bucket_size)) {}
};

}  // namespace chess
