/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

#include <chess/board.h>
#include <chess/types.h>
#include <feature/util.h>
#include <nnue/aligned_slice.h>
#include <nnue/aligned_scratchpad.h>
#include <nnue/dense_relu_affine_layer.h>
#include <nnue/feature_transformer.h>
#include <nnue/sparse_affine_layer.h>
#include <nnue/weights.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

template <typename T, std::size_t dim0, std::size_t dim1>
struct feature_transformer {
  const sparse_affine_layer<T, dim0, dim1>* weights_;

  aligned_slice<T, dim1> parent_slice_;
  aligned_slice<T, dim1> slice_;

  void clear() noexcept { slice_.copy_from(weights_->b); }
  void copy_parent() noexcept { slice_.copy_from(parent_slice_); }
  void insert(const std::size_t& idx) noexcept { weights_->insert_idx(idx, slice_); }
  void erase(const std::size_t& idx) noexcept { weights_->erase_idx(idx, slice_); }

  void copy_parent_insert_erase(const std::size_t& insert_idx, const std::size_t& erase_idx) noexcept {
    weights_->insert_erase_idx(insert_idx, erase_idx, parent_slice_, slice_);
  }

  void copy_parent_insert_erase_erase(const std::size_t& insert_idx, const std::size_t& erase_idx_0, const std::size_t& erase_idx_1) noexcept {
    weights_->insert_erase_erase_idx(insert_idx, erase_idx_0, erase_idx_1, parent_slice_, slice_);
  }

  feature_transformer(const sparse_affine_layer<T, dim0, dim1>* src, aligned_slice<T, dim1>&& parent_slice, aligned_slice<T, dim1>&& slice) noexcept
      : weights_{src}, parent_slice_{parent_slice}, slice_{slice} {}
};

}  // namespace nnue