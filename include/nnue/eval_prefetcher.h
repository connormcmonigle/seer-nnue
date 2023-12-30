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
#include <nnue/aligned_vector.h>
#include <nnue/dense_relu_affine_layer.h>
#include <nnue/feature_transformer.h>
#include <nnue/feature_transformer_prefetcher.h>
#include <nnue/sparse_affine_layer.h>
#include <nnue/weights.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

struct eval_prefetcher
    : public chess::
          sided<eval_prefetcher, feature_transformer_prefetcher<weights::quantized_parameter_type, feature::half_ka::numel, weights::base_dim>> {
  feature_transformer_prefetcher<weights::quantized_parameter_type, feature::half_ka::numel, weights::base_dim> white;
  feature_transformer_prefetcher<weights::quantized_parameter_type, feature::half_ka::numel, weights::base_dim> black;

  eval_prefetcher(const quantized_weights* src) noexcept : white{&src->shared}, black{&src->shared} {}
};

}  // namespace nnue