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
#include <nnue/dense_relu_affine_layer.h>
#include <nnue/sparse_affine_layer.h>
#include <nnue/weights_exporter.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <sstream>
#include <string>

namespace nnue {

struct weights {
  using parameter_type = float;
  using quantized_parameter_type = std::int16_t;
  using half_quantized_parameter_type = std::int8_t;

  static constexpr std::size_t base_dim = 768;

  static constexpr parameter_type shared_quantization_scale = static_cast<parameter_type>(512);
  static constexpr parameter_type fc0_weight_quantization_scale = static_cast<parameter_type>(1024);
  static constexpr parameter_type fc0_bias_quantization_scale = shared_quantization_scale * fc0_weight_quantization_scale;
  static constexpr parameter_type dequantization_scale = static_cast<parameter_type>(1) / (shared_quantization_scale * fc0_weight_quantization_scale);

  weights_streamer::signature_type signature_{0};

  sparse_affine_layer<parameter_type, feature::half_ka::numel, base_dim> shared{};
  dense_relu_affine_layer<2 * base_dim, 8, parameter_type> fc0{};

  dense_relu_affine_layer<8, 8, parameter_type> fc1{};
  dense_relu_affine_layer<16, 8, parameter_type> fc2{};
  dense_relu_affine_layer<24, 1, parameter_type> fc3{};

  [[nodiscard]] constexpr const weights_streamer::signature_type& signature() const noexcept { return signature_; }

  [[nodiscard]] constexpr std::size_t num_parameters() const noexcept {
    return shared.num_parameters() + fc0.num_parameters() + fc1.num_parameters() + fc2.num_parameters() + fc3.num_parameters();
  }

  template <typename Q>
  Q to() const noexcept {
    Q quantized{};

    quantized.signature_ = signature_;
    quantized.shared = shared.quantized<quantized_parameter_type>(shared_quantization_scale);
    
    quantized.fc0 =
        fc0.quantized<half_quantized_parameter_type, quantized_parameter_type>(fc0_weight_quantization_scale, fc0_bias_quantization_scale);

    quantized.white_fc0 = quantized.fc0;
    quantized.black_fc0 = quantized.white_fc0.half_input_flipped();

    quantized.fc1 = fc1;
    quantized.fc2 = fc2;
    quantized.fc3 = fc3;

    return quantized;
  }

  template <typename streamer_type>
  [[maybe_unused]] weights& load(streamer_type& streamer) noexcept {
    shared.load_(streamer);
    fc0.load_(streamer);
    fc1.load_(streamer);
    fc2.load_(streamer);
    fc3.load_(streamer);
    signature_ = streamer.signature();
    return *this;
  }

  [[maybe_unused]] weights& load(const std::string& path) noexcept {
    auto streamer = weights_streamer(path);
    return load(streamer);
  }
};

struct quantized_weights {
  using parameter_type = weights::parameter_type;
  using quantized_parameter_type = weights::quantized_parameter_type;
  using half_quantized_parameter_type = weights::half_quantized_parameter_type;

  static constexpr std::size_t base_dim = 768;

  weights_streamer::signature_type signature_{0};

  sparse_affine_layer<quantized_parameter_type, feature::half_ka::numel, base_dim> shared{};

  dense_relu_affine_layer<2 * base_dim, 8, half_quantized_parameter_type, quantized_parameter_type> fc0{};
  dense_relu_affine_layer<2 * base_dim, 8, half_quantized_parameter_type, quantized_parameter_type> white_fc0{};
  dense_relu_affine_layer<2 * base_dim, 8, half_quantized_parameter_type, quantized_parameter_type> black_fc0{};

  dense_relu_affine_layer<8, 8, parameter_type> fc1{};
  dense_relu_affine_layer<16, 8, parameter_type> fc2{};
  dense_relu_affine_layer<24, 1, parameter_type> fc3{};

  [[nodiscard]] constexpr const weights_streamer::signature_type& signature() const noexcept { return signature_; }

  [[nodiscard]] constexpr std::size_t num_parameters() const noexcept {
    return shared.num_parameters() + fc0.num_parameters() + fc1.num_parameters() + fc2.num_parameters() + fc3.num_parameters();
  }

  template <typename streamer_type>
  [[maybe_unused]] quantized_weights& load(streamer_type& streamer) noexcept {
    streamer.stream(&signature_);

    shared.load_(streamer);
    fc0.load_(streamer);
    fc1.load_(streamer);
    fc2.load_(streamer);
    fc3.load_(streamer);

    white_fc0 = fc0;
    black_fc0 = white_fc0.half_input_flipped();

    return *this;
  }

  template <typename exporter_type>
  [[maybe_unused]] const quantized_weights& write(exporter_type& exporter) const noexcept {
    exporter.write(&signature_);

    shared.write_(exporter);
    fc0.write_(exporter);
    fc1.write_(exporter);
    fc2.write_(exporter);
    fc3.write_(exporter);

    return *this;
  }

  [[maybe_unused]] quantized_weights& load(const std::string& path) noexcept {
    auto streamer = weights_streamer(path);
    return load(streamer);
  }

  [[maybe_unused]] const quantized_weights& write(const std::string& path) const noexcept {
    auto exporter = weights_exporter(path);
    return write(exporter);
  }
};

}  // namespace nnue
