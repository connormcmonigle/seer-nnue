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
#include <nnue/aligned_scratchpad.h>
#include <nnue/aligned_slice.h>
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
struct feature_transformer_prefetcher {
  const sparse_affine_layer<T, dim0, dim1>* weights_;

  void clear() noexcept {}
  void copy_parent() noexcept {}
  void insert(const std::size_t& idx) const noexcept { weights_->prefetch(idx); }
  void erase(const std::size_t& idx) const noexcept { weights_->prefetch(idx); }

  void copy_parent_insert_erase(const std::size_t& insert_idx, const std::size_t& erase_idx) noexcept {
    weights_->prefetch(insert_idx);
    weights_->prefetch(erase_idx);
  }

  void copy_parent_insert_erase_erase(const std::size_t& insert_idx, const std::size_t& erase_idx_0, const std::size_t& erase_idx_1) noexcept {
    weights_->prefetch(insert_idx);
    weights_->prefetch(erase_idx_0);
    weights_->prefetch(erase_idx_1);
  }

  feature_transformer_prefetcher(const sparse_affine_layer<T, dim0, dim1>* src) noexcept : weights_{src} {}
};

}  // namespace nnue
