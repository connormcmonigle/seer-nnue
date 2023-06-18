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
#include <chess_types.h>
#include <feature_util.h>
#include <nnue_util.h>
#include <search_constants.h>
#include <weights_streamer.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

struct weights {
  using parameter_type = float;
  using quantized_parameter_type = std::int16_t;

  static constexpr size_t base_dim = 768;

  static constexpr parameter_type shared_quantization_scale = static_cast<parameter_type>(512);
  static constexpr parameter_type fc0_weight_quantization_scale = static_cast<parameter_type>(1024);
  static constexpr parameter_type fc0_bias_quantization_scale = shared_quantization_scale * fc0_weight_quantization_scale;
  static constexpr parameter_type dequantization_scale = static_cast<parameter_type>(1) / (shared_quantization_scale * fc0_weight_quantization_scale);

  weights_streamer::signature_type signature_{0};
  big_affine<parameter_type, feature::half_ka::numel, base_dim> shared{};
  big_affine<quantized_parameter_type, feature::half_ka::numel, base_dim> quantized_shared{};

  stack_relu_affine<parameter_type, 2 * base_dim, 8> fc0{};
  stack_relu_affine<quantized_parameter_type, 2 * base_dim, 8> white_quantized_fc0{};
  stack_relu_affine<quantized_parameter_type, 2 * base_dim, 8> black_quantized_fc0{};

  stack_relu_affine<parameter_type, 8, 8> fc1{};
  stack_relu_affine<parameter_type, 16, 8> fc2{};
  stack_relu_affine<parameter_type, 24, 1> fc3{};

  size_t signature() const { return signature_; }

  size_t num_parameters() const {
    return shared.num_parameters() + fc0.num_parameters() + fc1.num_parameters() + fc2.num_parameters() + fc3.num_parameters();
  }

  template <typename streamer_type>
  weights& load(streamer_type& ws) {
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

  weights& load(const std::string& path) {
    auto ws = weights_streamer(path);
    return load(ws);
  }
};

template <typename T>
struct feature_transformer {
  const big_affine<T, feature::half_ka::numel, weights::base_dim>* weights_;

  aligned_slice<T, weights::base_dim> parent_slice_;
  aligned_slice<T, weights::base_dim> slice_;

  void clear() { slice_.copy_from(weights_->b); }
  void copy_parent() { slice_.copy_from(parent_slice_); }
  void insert(const size_t& idx) { weights_->insert_idx(idx, slice_); }
  void erase(const size_t& idx) { weights_->erase_idx(idx, slice_); }

  void copy_parent_insert_erase(const size_t& insert_idx, const size_t& erase_idx) {
    weights_->insert_erase_idx(insert_idx, erase_idx, parent_slice_, slice_);
  }

  void copy_parent_insert_erase_erase(const size_t& insert_idx, const size_t& erase_idx_0, const size_t& erase_idx_1) {
    weights_->insert_erase_erase_idx(insert_idx, erase_idx_0, erase_idx_1, parent_slice_, slice_);
  }

  feature_transformer(
      const big_affine<T, feature::half_ka::numel, weights::base_dim>* src,
      aligned_slice<T, weights::base_dim>&& parent_slice,
      aligned_slice<T, weights::base_dim>&& slice)
      : weights_{src}, parent_slice_{parent_slice}, slice_{slice} {}
};

struct eval : chess::sided<eval, feature_transformer<weights::quantized_parameter_type>> {
  static constexpr size_t base_dim = weights::base_dim;
  static constexpr size_t feature_transformer_dim = 2 * base_dim;
  static constexpr size_t scratchpad_depth = 256;

  using parameter_type = weights::parameter_type;
  using quantized_parameter_type = weights::quantized_parameter_type;
  using scratchpad_type = stack_scratchpad<quantized_parameter_type, scratchpad_depth * feature_transformer_dim>;

  const weights* weights_;
  scratchpad_type* scratchpad_;

  size_t scratchpad_idx_;

  aligned_slice<quantized_parameter_type, feature_transformer_dim> parent_base_;
  aligned_slice<quantized_parameter_type, feature_transformer_dim> base_;

  feature_transformer<quantized_parameter_type> white;
  feature_transformer<quantized_parameter_type> black;

  inline parameter_type propagate(const bool pov) const {
    const auto x1 = (pov ? weights_->white_quantized_fc0 : weights_->black_quantized_fc0)
                        .forward(base_)
                        .dequantized<parameter_type>(weights::dequantization_scale);

    const auto x2 = splice(x1, weights_->fc1.forward(x1));
    const auto x3 = splice(x2, weights_->fc2.forward(x2));
    return weights_->fc3.forward(x3).item();
  }

  inline search::score_type evaluate(const bool pov, const parameter_type& phase) const {
    constexpr parameter_type one = static_cast<parameter_type>(1.0);
    constexpr parameter_type mg = static_cast<parameter_type>(1.1);
    constexpr parameter_type eg = static_cast<parameter_type>(0.7);

    const parameter_type prediction = propagate(pov);
    const parameter_type eval = phase * mg * prediction + (one - phase) * eg * prediction;

    const parameter_type value =
        search::logit_scale<parameter_type> * std::clamp(eval, search::min_logit<parameter_type>, search::max_logit<parameter_type>);
    return static_cast<search::score_type>(value);
  }

  eval next_child() const {
    const size_t next_scratchpad_idx = scratchpad_idx_ + 1;
    return eval(weights_, scratchpad_, scratchpad_idx_, next_scratchpad_idx);
  }

  eval(const weights* src, scratchpad_type* scratchpad, const size_t& parent_scratchpad_idx, const size_t& scratchpad_idx)
      : weights_{src},
        scratchpad_{scratchpad},
        scratchpad_idx_{scratchpad_idx},
        parent_base_(scratchpad_->get_nth_slice<feature_transformer_dim>(parent_scratchpad_idx)),
        base_(scratchpad_->get_nth_slice<feature_transformer_dim>(scratchpad_idx_)),
        white{&src->quantized_shared, parent_base_.slice<base_dim>(), base_.slice<base_dim>()},
        black{&src->quantized_shared, parent_base_.slice<base_dim, base_dim>(), base_.slice<base_dim, base_dim>()} {}
};

}  // namespace nnue
