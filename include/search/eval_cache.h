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
  using persisted_eval_type = std::int16_t;

  zobrist::half_hash_type hash_{};
  zobrist::quarter_hash_type eval_feature_hash_{};
  persisted_eval_type persisted_eval_{};

  [[nodiscard]] constexpr const zobrist::half_hash_type& hash() const noexcept { return hash_; }
  [[nodiscard]] constexpr const zobrist::quarter_hash_type& eval_feature_hash() const noexcept { return eval_feature_hash_; }

  [[nodiscard]] constexpr score_type eval() const noexcept { return static_cast<score_type>(persisted_eval_); }

  [[nodiscard]] static constexpr eval_cache_entry
  make(const zobrist::hash_type& hash, const zobrist::quarter_hash_type& eval_feature_hash, const score_type& eval) noexcept {
    const zobrist::half_hash_type hash_upper_half = zobrist::upper_half(hash);
    const persisted_eval_type persisted_eval = static_cast<persisted_eval_type>(eval);

    return eval_cache_entry{hash_upper_half, eval_feature_hash, persisted_eval};
  }
};

struct eval_cache {
  static constexpr std::size_t size_mb = 8;
  static constexpr std::size_t N = (size_mb << 20) / sizeof(eval_cache_entry);
  static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");

  std::array<eval_cache_entry, N> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const zobrist::hash_type& hash) noexcept { return hash & (N - 1); }
  inline void prefetch(const zobrist::hash_type& hash) const noexcept { __builtin_prefetch(data.data() + hash_function(hash)); }

  [[nodiscard]] constexpr std::optional<eval_cache_entry> find(const zobrist::hash_type& hash) const noexcept {
    if (data[hash_function(hash)].hash() == zobrist::upper_half(hash)) { return data[hash_function(hash)]; }
    return std::nullopt;
  }

  void insert(const zobrist::hash_type& hash, const eval_cache_entry& entry) noexcept { data[hash_function(hash)] = entry; }

  void clear() noexcept { return data.fill(eval_cache_entry{}); }
};

}  // namespace search
