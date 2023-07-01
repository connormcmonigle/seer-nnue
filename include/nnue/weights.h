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
#include <nnue/sparse_affine_layer.h>
#include <nnue/dense_relu_affine_layer.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

struct weights {
  using parameter_type = float;
  using quantized_parameter_type = std::int16_t;

  static constexpr std::size_t base_dim = 768;

  static constexpr parameter_type shared_quantization_scale = static_cast<parameter_type>(512);
  static constexpr parameter_type fc0_weight_quantization_scale = static_cast<parameter_type>(1024);
  static constexpr parameter_type fc0_bias_quantization_scale = shared_quantization_scale * fc0_weight_quantization_scale;
  static constexpr parameter_type dequantization_scale = static_cast<parameter_type>(1) / (shared_quantization_scale * fc0_weight_quantization_scale);

  weights_streamer::signature_type signature_{0};
  
  sparse_affine_layer<parameter_type, feature::half_ka::numel, base_dim> shared{};
  sparse_affine_layer<quantized_parameter_type, feature::half_ka::numel, base_dim> quantized_shared{};

  dense_relu_affine_layer<parameter_type, 2 * base_dim, 8> fc0{};
  dense_relu_affine_layer<quantized_parameter_type, 2 * base_dim, 8> white_quantized_fc0{};
  dense_relu_affine_layer<quantized_parameter_type, 2 * base_dim, 8> black_quantized_fc0{};

  dense_relu_affine_layer<parameter_type, 8, 8> fc1{};
  dense_relu_affine_layer<parameter_type, 16, 8> fc2{};
  dense_relu_affine_layer<parameter_type, 24, 1> fc3{};

  [[nodiscard]] constexpr const weights_streamer::signature_type& signature() const noexcept { return signature_; }

  [[nodiscard]] constexpr std::size_t num_parameters() const noexcept {
    return shared.num_parameters() + fc0.num_parameters() + fc1.num_parameters() + fc2.num_parameters() + fc3.num_parameters();
  }

  template <typename streamer_type>
  [[maybe_unused]] weights& load(streamer_type& ws) noexcept {
    shared.load_(ws);
    fc0.load_(ws);
    fc1.load_(ws);
    fc2.load_(ws);
    fc3.load_(ws);
    signature_ = ws.signature();

    quantized_shared = shared.quantized<quantized_parameter_type>(shared_quantization_scale);
    white_quantized_fc0 = fc0.quantized<quantized_parameter_type>(fc0_weight_quantization_scale, fc0_bias_quantization_scale);
    black_quantized_fc0 = white_quantized_fc0.half_input_flipped();

    return *this;
  }

  [[maybe_unused]] weights& load(const std::string& path) noexcept {
    auto ws = weights_streamer(path);
    return load(ws);
  }
};

}  // namespace nnue