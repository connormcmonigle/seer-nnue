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

#include <nnue_model.h>
#include <nnue_util.h>
#include <search_constants.h>
#include <zobrist_util.h>

#include <array>
#include <optional>

namespace nnue {

struct feature_transformer_cache {
  static constexpr size_t size_mb = 12;
  static constexpr size_t N = (size_mb << 20) / (eval::base_dim * sizeof(eval::quantized_parameter_type));

  std::array<zobrist::hash_type, N> keys{};
  stack_scratchpad<eval::quantized_parameter_type, N * eval::base_dim> scratchpad_;

  constexpr size_t hash_function(const zobrist::hash_type& hash) const { return hash % keys.size(); }

  void prefetch(const zobrist::hash_type& hash) {
    const size_t idx = hash_function(hash);
    __builtin_prefetch(scratchpad_.get_nth_slice<eval::base_dim>(idx).data);
  }

  std::optional<eval::base_type> find(const zobrist::hash_type& hash) {
    if (const size_t idx = hash_function(hash); keys[idx] == hash) { return scratchpad_.get_nth_slice<eval::base_dim>(idx); }
    return std::nullopt;
  }

  void insert(const zobrist::hash_type& hash, const eval::base_type& state) {
    const size_t idx = hash_function(hash);
    keys[idx] = hash;
    scratchpad_.get_nth_slice<eval::base_dim>(idx).copy_from(state.data);
  }

  void clear() { return keys.fill(zobrist::hash_type{}); }
};

}  // namespace nnue