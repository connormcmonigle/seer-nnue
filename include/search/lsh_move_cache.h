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
#include <chess/move_list.h>
#include <chess/types.h>
#include <search/search_constants.h>
#include <zobrist/util.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <random>
#include <utility>

namespace search {

namespace detail {

constexpr std::mt19937::result_type seed = 0x019ec6dc;
using index_type = std::uint16_t;

}  // namespace detail

namespace lsh {

template <std::size_t S>
struct lsh_signature {
  static constexpr std::size_t size = S;

  std::array<detail::index_type, S> signature_;

  [[nodiscard]] std::size_t hash_value() const noexcept {
    std::size_t value{144451};
    auto hash_function = std::hash<detail::index_type>{};
    for (std::size_t i(0); i < S; ++i) { value = (value << 1) ^ hash_function(signature_[i]); }
    return value;
  }

  [[nodiscard]] const detail::index_type& at(const std::size_t& idx) const noexcept { return signature_[idx]; }
  [[nodiscard]] detail::index_type& at(const std::size_t& idx) noexcept { return signature_[idx]; }

  static constexpr lsh_signature of(const detail::index_type& value) {
    lsh_signature result{};
    result.signature_.fill(value);
    return result;
  }
};

template <std::size_t N, std::size_t S>
struct lsh_min_hasher {
  using lsh_signature_type = lsh_signature<S>;
  static constexpr std::size_t cardinality = N;

  std::array<detail::index_type, N * S> encoding_values_;

  template <typename F>
  [[nodiscard]] lsh_signature_type encode(F&& indicator_function) const noexcept {
    auto signature = lsh_signature_type::of(N);

    auto update_signature = [this, &signature](const std::size_t& idx) {
      for (std::size_t i(0); i < S; ++i) {
        const detail::index_type encoded_value = encoding_values_[i * N + idx];
        signature.at(i) = std::min(signature.at(i), encoded_value);
      }
    };

    for (std::size_t idx(0); idx < N; ++idx) {
      if (indicator_function(idx)) { update_signature(idx); }
    }

    return signature;
  }

  lsh_min_hasher() noexcept {
    std::mt19937 generator(detail::seed);

    for (std::size_t i(0); i < S; ++i) {
      const auto begin = encoding_values_.begin() + i * N;
      std::iota(begin, begin + N, detail::index_type{});
      std::shuffle(begin, begin + N, generator);
    }
  }
};

}  // namespace lsh

struct lsh_move_cache_entry {
  static constexpr std::size_t moves_per_entry = 4;

  std::array<chess::move, moves_per_entry> data_{};

  void insert(const chess::move& mv) noexcept {
    if (std::find(data_.begin(), data_.end(), mv) != data_.end()) { return; }

    if (const auto dst_iter = std::find(data_.begin(), data_.end(), chess::move::null()); dst_iter != data_.end()) {
      *dst_iter = mv;
      return;
    }

    const std::size_t entry_idx = chess::move_hash{}(mv) % moves_per_entry;
    data_[entry_idx] = mv;
  }

  [[nodiscard]] chess::move move_for(const chess::move_list& list) const noexcept {
    const auto iter = std::find_if(data_.begin(), data_.end(), [&list](const chess::move& mv) { return list.has(mv); });
    return iter != data_.end() ? *iter : chess::move::null();
  }

  lsh_move_cache_entry() { data_.fill(chess::move::null()); }
};

struct sided_lsh_move_cache_entry : public chess::sided<sided_lsh_move_cache_entry, lsh_move_cache_entry> {
  lsh_move_cache_entry white;
  lsh_move_cache_entry black;
  sided_lsh_move_cache_entry() noexcept : white{}, black{} {}
};

template <typename H, std::size_t N>
struct lsh_move_cache {
  using lsh_hasher_type = H;

  H hasher_;
  std::array<sided_lsh_move_cache_entry, N> entries_;

  template <typename F>
  [[nodiscard]] std::size_t hash_function_(F&& indicator_function) {
    return hasher_.encode(std::forward<F>(indicator_function)).hash_value() % N;
  }

  template <typename F>
  [[nodiscard]] const sided_lsh_move_cache_entry* at(F&& indicator_function) const noexcept {
    return entries_.data() + hash_function_(std::forward<F>(indicator_function));
  }

  template <typename F>
  [[nodiscard]] sided_lsh_move_cache_entry* at(F&& indicator_function) noexcept {
    return entries_.data() + hash_function_(std::forward<F>(indicator_function));;
    ;
  }

  void reset() noexcept { entries_.fill(sided_lsh_move_cache_entry{}); }
  lsh_move_cache() noexcept { entries_.fill(sided_lsh_move_cache_entry{}); }
};

}  // namespace search