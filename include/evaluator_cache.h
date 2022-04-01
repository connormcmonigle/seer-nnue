/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
  the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not,
  see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <board.h>
#include <eval_cache.h>
#include <feature_util.h>
#include <nnue_model.h>
#include <nnue_util.h>
#include <search_constants.h>
#include <zobrist_util.h>

#include <array>
#include <optional>

namespace search {

struct evaluator_cache {
  using half_vector_type = nnue::stack_vector<nnue::weights::parameter_type, nnue::weights::base_dim>;

  static constexpr size_t size_mb = 16;
  static constexpr size_t N = (size_mb << 19) / sizeof(half_vector_type);

  const nnue::weights* weights_{nullptr};

  std::array<zobrist::hash_type, N> keys{};
  half_vector_type white[N];
  half_vector_type black[N];

  constexpr size_t hash_function(const zobrist::hash_type& key) const { return key % N; }

  void prefetch(const zobrist::hash_type& key) const {
    __builtin_prefetch(white + hash_function(key));
    __builtin_prefetch(black + hash_function(key));
  }

  bool contains(const zobrist::hash_type& key) const { return keys[hash_function(key)] == key; }

  nnue::eval get(const zobrist::hash_type& key) const {
    const size_t hash = hash_function(key);
    return nnue::eval(weights_, white[hash], black[hash]);
  }

  void insert(const zobrist::hash_type& key, const nnue::eval& eval) {
    const size_t hash = hash_function(key);
    keys[hash] = key;
    white[hash] = eval.white.active();
    black[hash] = eval.black.active();
  }

  void clear() { keys.fill(zobrist::hash_type{}); }
  void set_weights(const nnue::weights* weights) { weights_ = weights; }
};

}  // namespace search