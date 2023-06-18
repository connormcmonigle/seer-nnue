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

namespace search {

constexpr size_t cache_line_size = 64;
enum class bound_type { upper, lower, exact };

struct transposition_table_entry {
  static constexpr zobrist::hash_type empty_key = zobrist::hash_type{};
  static constexpr size_t gen_bits = 7;
  using gen_type = std::uint8_t;

  using bound_ = bit::range<bound_type, 0, 2>;
  using score_ = bit::next_range<bound_, std::int16_t>;
  using best_move_ = bit::next_range<score_, chess::move::data_type, chess::move::width>;
  using depth_ = bit::next_range<best_move_, std::uint8_t>;
  using gen_ = bit::next_range<depth_, gen_type, gen_bits>;
  using was_exact_or_lb_ = bit::next_flag<gen_>;

  zobrist::hash_type key_{empty_key};
  zobrist::hash_type value_{};

  zobrist::hash_type key() const { return key_ ^ value_; }

  bound_type bound() const { return bound_::get(value_); }
  score_type score() const { return static_cast<score_type>(score_::get(value_)); }
  gen_type gen() const { return gen_::get(value_); }
  depth_type depth() const { return static_cast<depth_type>(depth_::get(value_)); }
  chess::move best_move() const { return chess::move{best_move_::get(value_)}; }
  bool was_exact_or_lb() const { return was_exact_or_lb_::get(value_); }

  bool is_empty() const { return key_ == empty_key; }

  bool is_current(const gen_type& gen) const { return gen == gen_::get(value_); }

  transposition_table_entry& set_gen(const gen_type& gen) {
    key_ ^= value_;
    gen_::set(value_, gen);
    key_ ^= value_;
    return *this;
  }

  transposition_table_entry& merge(const transposition_table_entry& other) {
    if (bound() == bound_type::upper && other.was_exact_or_lb() && key() == other.key()) {
      key_ ^= value_;
      best_move_::set(value_, other.best_move().data);
      was_exact_or_lb_::set(value_, true);
      key_ ^= value_;
    }

    return *this;
  }

  transposition_table_entry(
      const zobrist::hash_type& key, const bound_type& bound, const score_type& score, const chess::move& mv, const depth_type& depth)
      : key_{key} {
    bound_::set(value_, bound);
    score_::set(value_, static_cast<score_::type>(score));
    best_move_::set(value_, mv.data);
    depth_::set(value_, static_cast<depth_::type>(depth));
    was_exact_or_lb_::set(value_, bound != bound_type::upper);
    key_ ^= value_;
  }

  transposition_table_entry() {}
};

template <size_t N>
struct alignas(cache_line_size) bucket {
  transposition_table_entry data[N];

  std::optional<transposition_table_entry> match(const transposition_table_entry::gen_type& gen, const zobrist::hash_type& key) {
    for (auto& elem : data) {
      if (elem.key() == key) { return std::optional(elem.set_gen(gen)); }
    }
    return std::nullopt;
  }

  transposition_table_entry* to_replace(const transposition_table_entry::gen_type& gen, const zobrist::hash_type& key) {
    auto worst = std::begin(data);
    for (auto iter = std::begin(data); iter != std::end(data); ++iter) {
      if (iter->key() == key) { return iter; }

      const bool is_worse = (!iter->is_current(gen) && worst->is_current(gen)) || (iter->is_empty() && !worst->is_empty()) ||
                            ((iter->is_current(gen) == worst->is_current(gen)) && (iter->depth() < worst->depth()));

      if (is_worse) { worst = iter; }
    }

    return worst;
  }
};

struct transposition_table {
  static constexpr size_t per_bucket = cache_line_size / sizeof(transposition_table_entry);
  static constexpr size_t one_mb = (1 << 20) / cache_line_size;

  using bucket_type = bucket<per_bucket>;

  static_assert(cache_line_size % sizeof(transposition_table_entry) == 0, "transposition_table_entry must divide cache_line_size");
  static_assert(sizeof(bucket_type) == cache_line_size && alignof(bucket_type) == cache_line_size, "bucket_type must be cache_line_size aligned");

  std::atomic<transposition_table_entry::gen_type> current_gen{0};
  std::vector<bucket_type> data;

  void prefetch(const zobrist::hash_type& key) const { __builtin_prefetch(data.data() + hash_function(key)); }

  void clear() {
    for (auto& elem : data) { elem = bucket_type{}; }
  }

  void resize(size_t size) {
    clear();
    data.resize(size * one_mb, bucket_type{});
  }

  void update_gen() {
    using gen_type = transposition_table_entry::gen_type;
    constexpr gen_type limit = gen_type{1} << transposition_table_entry::gen_bits;
    current_gen = (current_gen + 1) % limit;
  }

  size_t hash_function(const zobrist::hash_type& hash) const { return hash % data.size(); }

  __attribute__((no_sanitize("thread"))) transposition_table& insert(const transposition_table_entry& entry) {
    constexpr depth_type offset = 2;
    const transposition_table_entry::gen_type gen = current_gen.load(std::memory_order_relaxed);

    transposition_table_entry* to_replace = data[hash_function(entry.key())].to_replace(gen, entry.key());

    const bool should_replace =
        (entry.bound() == bound_type::exact) || (entry.key() != to_replace->key()) || ((entry.depth() + offset) >= to_replace->depth());

    if (should_replace) { *to_replace = transposition_table_entry(entry).set_gen(gen).merge(*to_replace); }

    return *this;
  }

  __attribute__((no_sanitize("thread"))) std::optional<transposition_table_entry> find(const zobrist::hash_type& key) {
    const transposition_table_entry::gen_type gen = current_gen.load(std::memory_order_relaxed);
    return data[hash_function(key)].match(gen, key);
  }

  transposition_table(size_t size) : data(size * one_mb) {}
};

}  // namespace search
