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

#include <chess/move.h>
#include <search/search_constants.h>
#include <util/bit_range.h>
#include <zobrist/util.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <string_view>
#include <vector>

namespace search {

constexpr std::size_t cache_line_size = 64;
enum class bound_type { upper, lower, exact };

struct transposition_table_entry {
  static constexpr zobrist::hash_type empty_key = zobrist::hash_type{};
  static constexpr std::size_t gen_bits = 6;
  using gen_type = std::uint8_t;

  using bound_ = util::bit_range<bound_type, 0, 2>;
  using score_ = util::next_bit_range<bound_, std::int16_t>;
  using best_move_ = util::next_bit_range<score_, chess::move::data_type, chess::move::width>;
  using depth_ = util::next_bit_range<best_move_, std::uint8_t>;
  using gen_ = util::next_bit_range<depth_, gen_type, gen_bits>;
  using tt_pv_ = util::next_bit_flag<gen_>;
  using was_exact_or_lb_ = util::next_bit_flag<tt_pv_>;

  zobrist::hash_type key_{empty_key};
  zobrist::hash_type value_{};

  [[nodiscard]] constexpr zobrist::hash_type key() const noexcept { return key_ ^ value_; }

  [[nodiscard]] constexpr bound_type bound() const noexcept { return bound_::get(value_); }
  [[nodiscard]] constexpr score_type score() const noexcept { return static_cast<score_type>(score_::get(value_)); }
  [[nodiscard]] constexpr gen_type gen() const noexcept { return gen_::get(value_); }
  [[nodiscard]] constexpr depth_type depth() const noexcept { return static_cast<depth_type>(depth_::get(value_)); }
  [[nodiscard]] constexpr chess::move best_move() const noexcept { return chess::move{best_move_::get(value_)}; }

  [[nodiscard]] constexpr bool was_exact_or_lb() const noexcept { return was_exact_or_lb_::get(value_); }
  [[nodiscard]] constexpr bool tt_pv() const noexcept { return tt_pv_::get(value_); }

  [[nodiscard]] constexpr bool is_empty() const noexcept { return key_ == empty_key; }
  [[nodiscard]] constexpr bool is_current(const gen_type& gen) const noexcept { return gen == gen_::get(value_); }

  [[maybe_unused]] constexpr transposition_table_entry& set_gen(const gen_type& gen) noexcept {
    key_ ^= value_;
    gen_::set(value_, gen);
    key_ ^= value_;
    return *this;
  }

  [[maybe_unused]] constexpr transposition_table_entry& merge(const transposition_table_entry& other) noexcept {
    if (bound() == bound_type::upper && other.was_exact_or_lb() && key() == other.key()) {
      key_ ^= value_;
      best_move_::set(value_, other.best_move().data);
      was_exact_or_lb_::set(value_, true);
      key_ ^= value_;
    }

    return *this;
  }

  constexpr transposition_table_entry(
      const zobrist::hash_type& key,
      const bound_type& bound,
      const score_type& score,
      const chess::move& mv,
      const depth_type& depth,
      const bool& tt_pv = false) noexcept
      : key_{key} {
    bound_::set(value_, bound);
    score_::set(value_, static_cast<score_::type>(score));

    best_move_::set(value_, mv.data);
    depth_::set(value_, static_cast<depth_::type>(depth));

    tt_pv_::set(value_, tt_pv);
    was_exact_or_lb_::set(value_, bound != bound_type::upper);
    key_ ^= value_;
  }

  constexpr transposition_table_entry() noexcept = default;
};

template <std::size_t N>
struct alignas(cache_line_size) bucket {
  transposition_table_entry data[N];

  [[nodiscard]] constexpr std::optional<transposition_table_entry> match(
      const transposition_table_entry::gen_type& gen, const zobrist::hash_type& key) noexcept {
    for (auto& elem : data) {
      if (elem.key() == key) { return std::optional(elem.set_gen(gen)); }
    }

    return std::nullopt;
  }

  [[nodiscard]] constexpr transposition_table_entry* to_replace(
      const transposition_table_entry::gen_type& gen, const zobrist::hash_type& key) noexcept {
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
  static constexpr std::size_t per_bucket = cache_line_size / sizeof(transposition_table_entry);
  static constexpr std::size_t one_mb = (1 << 20) / cache_line_size;

  using bucket_type = bucket<per_bucket>;

  static_assert(cache_line_size % sizeof(transposition_table_entry) == 0, "transposition_table_entry must divide cache_line_size");
  static_assert(sizeof(bucket_type) == cache_line_size && alignof(bucket_type) == cache_line_size, "bucket_type must be cache_line_size aligned");

  std::atomic<transposition_table_entry::gen_type> current_gen{0};
  std::vector<bucket_type> data;

  [[nodiscard]] inline std::size_t hash_function(const zobrist::hash_type& hash) const noexcept { return hash % data.size(); }
  inline void prefetch(const zobrist::hash_type& key) const noexcept { __builtin_prefetch(data.data() + hash_function(key)); }

  void clear() noexcept;
  void resize(const std::size_t& size) noexcept;
  void update_gen() noexcept;

  // __attribute__((no_sanitize("thread")))
  [[maybe_unused]] transposition_table& insert(const transposition_table_entry& entry) noexcept;

  // __attribute__((no_sanitize("thread")))
  [[nodiscard]] std::optional<transposition_table_entry> find(const zobrist::hash_type& key) noexcept;

  explicit transposition_table(const std::size_t& size) noexcept : data(size * one_mb) {}
};

}  // namespace search
