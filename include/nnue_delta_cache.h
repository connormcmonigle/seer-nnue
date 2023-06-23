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

#include <nnue_util.h>
#include <zobrist_util.h>

namespace nnue {

template <typename T, size_t dim>
struct delta_cache {
  static constexpr size_t N = 8192;
  static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");

  stack_scratchpad<T, N * dim> scratch_pad_{};
  zobrist::hash_type keys_[N]{};

  constexpr size_t hash_function(const zobrist::hash_type& hash) const { return hash & (N - 1); }

  bool cache_hit(const zobrist::hash_type& hash) const { return hash == keys_[hash_function(hash)]; }

  aligned_slice<T, dim> insert_and_fetch(const zobrist::hash_type& hash) {
    const size_t idx = hash_function(hash);
    
    keys_[idx] = hash;
    return scratch_pad_.template get_nth_slice<dim>(idx);
  }
};

}  // namespace nnue