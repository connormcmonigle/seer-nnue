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
#include <nnue/sparse_affine_layer.h>
#include <nnue/weights.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

struct eval : public chess::sided<eval, feature_transformer<weights::quantized_parameter_type, feature::half_ka::numel, weights::base_dim>> {
  static constexpr std::size_t base_dim = weights::base_dim;
  static constexpr std::size_t feature_transformer_dim = 2 * base_dim;
  static constexpr std::size_t scratchpad_depth = 256;

  using parameter_type = weights::parameter_type;
  using quantized_parameter_type = weights::quantized_parameter_type;
  using scratchpad_type = aligned_scratchpad<quantized_parameter_type, scratchpad_depth * feature_transformer_dim>;

  const quantized_weights* weights_;
  scratchpad_type* scratchpad_;

  std::size_t scratchpad_idx_;

  aligned_slice<quantized_parameter_type, feature_transformer_dim> parent_base_;
  aligned_slice<quantized_parameter_type, feature_transformer_dim> base_;

  feature_transformer<quantized_parameter_type, feature::half_ka::numel, weights::base_dim> white;
  feature_transformer<quantized_parameter_type, feature::half_ka::numel, weights::base_dim> black;

  [[nodiscard]] bool is_hot(const std::size_t& idx) const noexcept { return base_.data[idx] > quantized_parameter_type{}; }

  [[nodiscard]] inline parameter_type propagate(const bool pov) const noexcept {
    const auto x1 = (pov ? weights_->white_fc0 : weights_->black_fc0)
                        .forward(base_)
                        .dequantized<parameter_type>(weights::dequantization_scale);

    const auto x2 = concat(x1, weights_->fc1.forward(x1));
    const auto x3 = concat(x2, weights_->fc2.forward(x2));
    return weights_->fc3.forward(x3).item();
  }

  [[nodiscard]] inline search::score_type evaluate(const bool pov, const parameter_type& phase) const noexcept {
    constexpr auto one = static_cast<parameter_type>(1.0);
    constexpr auto mg = static_cast<parameter_type>(1.1);
    constexpr auto eg = static_cast<parameter_type>(0.7);

    const parameter_type prediction = propagate(pov);
    const parameter_type eval = phase * mg * prediction + (one - phase) * eg * prediction;

    const parameter_type value =
        search::logit_scale<parameter_type> * std::clamp(eval, search::min_logit<parameter_type>, search::max_logit<parameter_type>);
    return static_cast<search::score_type>(value);
  }

  [[nodiscard]] eval next_child() const noexcept {
    const std::size_t next_scratchpad_idx = scratchpad_idx_ + 1;
    return eval(weights_, scratchpad_, scratchpad_idx_, next_scratchpad_idx);
  }

  eval(const quantized_weights* src, scratchpad_type* scratchpad, const std::size_t& parent_scratchpad_idx, const std::size_t& scratchpad_idx) noexcept
      : weights_{src},
        scratchpad_{scratchpad},
        scratchpad_idx_{scratchpad_idx},
        parent_base_(scratchpad_->get_nth_slice<feature_transformer_dim>(parent_scratchpad_idx)),
        base_(scratchpad_->get_nth_slice<feature_transformer_dim>(scratchpad_idx_)),
        white{&src->shared, parent_base_.slice<base_dim>(), base_.slice<base_dim>()},
        black{&src->shared, parent_base_.slice<base_dim, base_dim>(), base_.slice<base_dim, base_dim>()} {}
};

}  // namespace nnue
