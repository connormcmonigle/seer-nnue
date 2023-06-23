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

#include <chess_types.h>
#include <feature_util.h>
#include <move.h>
#include <square.h>
#include <zobrist_util.h>

#include <algorithm>

namespace feature {
namespace half_ka {

template <size_t N>
struct feature_zobrist_src {
  std::array<zobrist::hash_type, N> data_{};

  zobrist::hash_type get(const size_t& feature_index) const { return data_[feature_index]; }

  feature_zobrist_src() {
    std::transform(data_.begin(), data_.end(), data_.begin(), [](auto...) { return zobrist::random_bit_string(); });
  }
};

struct delta_zobrist_sources {
  static inline const feature_zobrist_src<numel> insert_src{};
  static inline const feature_zobrist_src<numel> erase_src{};
  delta_zobrist_sources() = delete;
};

zobrist::hash_type delta_hash(const size_t& insert_idx, const size_t& erase_idx) {
  return delta_zobrist_sources::insert_src.get(insert_idx) ^ delta_zobrist_sources::erase_src.get(erase_idx);
}

zobrist::hash_type delta_hash(const size_t& insert_idx, const size_t& erase_idx_0, const size_t& erase_idx_1) {
  return delta_zobrist_sources::insert_src.get(insert_idx) ^ delta_zobrist_sources::erase_src.get(erase_idx_0) ^
         delta_zobrist_sources::erase_src.get(erase_idx_1);
  ;
}

}  // namespace half_ka
}  // namespace feature