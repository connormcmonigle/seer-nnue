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

#include <enum_util.h>
#include <nnue_util.h>
#include <search_constants.h>
#include <weights_streamer.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

constexpr size_t half_ka_numel = 768 * 64;
constexpr size_t max_active_half_features = 32;
constexpr size_t base_dim = 160;

template <typename T, typename Q = std::int16_t>
struct weights {
  static_assert(std::is_floating_point_v<T>);
  static_assert(std::is_integral_v<Q>);

  static constexpr T fixed_wb_quantization_mul = static_cast<T>(255);
  static constexpr T fixed_fc0_quantization_mul = static_cast<T>(127);
  static constexpr T fixed_dequantization_mul = static_cast<T>(1) / (fixed_wb_quantization_mul * fixed_fc0_quantization_mul);

  typename weights_streamer<T>::signature_type signature_{0};
  big_affine<T, half_ka_numel, base_dim> w{};
  big_affine<T, half_ka_numel, base_dim> b{};
  stack_affine<T, 2 * base_dim, 16> fc0{};

  big_affine<Q, half_ka_numel, base_dim> quant_w{};
  big_affine<Q, half_ka_numel, base_dim> quant_b{};
  stack_affine<Q, 2 * base_dim, 16> quant_fc0{};

  stack_affine<T, 16, 16> fc1{};
  stack_affine<T, 32, 16> fc2{};
  stack_affine<T, 48, 1> fc3{};

  size_t signature() const { return signature_; }

  size_t num_parameters() const {
    return w.num_parameters() + b.num_parameters() + fc0.num_parameters() + fc1.num_parameters() + fc2.num_parameters() + fc3.num_parameters();
  }

  template <typename streamer_type>
  weights<T>& load(streamer_type& ws) {
    w.load_(ws);
    b.load_(ws);
    fc0.load_(ws);

    quantized_affine(w, quant_w, fixed_wb_quantization_mul);
    quantized_affine(b, quant_b, fixed_wb_quantization_mul);
    quantized_affine(fc0, quant_fc0, fixed_fc0_quantization_mul);

    fc1.load_(ws);
    fc2.load_(ws);
    fc3.load_(ws);
    signature_ = ws.signature();
    return *this;
  }

  weights<T>& load(const std::string& path) {
    auto ws = weights_streamer<T>(path);
    return load(ws);
  }
};

template <typename T>
struct feature_transformer {
  const big_affine<T, half_ka_numel, base_dim>* weights_;
  stack_vector<T, base_dim> active_;
  constexpr stack_vector<T, base_dim> active() const { return active_; }

  void clear() { active_ = stack_vector<T, base_dim>::from(weights_->b); }
  void insert(const size_t idx) { weights_->insert_idx(idx, active_); }
  void erase(const size_t idx) { weights_->erase_idx(idx, active_); }

  feature_transformer(const big_affine<T, half_ka_numel, base_dim>* src) : weights_{src} { clear(); }
};

template <typename T, typename Q = std::int16_t>
struct eval : chess::sided<eval<T, Q>, feature_transformer<Q>> {
  const weights<T, Q>* weights_;
  feature_transformer<Q> white;
  feature_transformer<Q> black;

  inline T propagate(const bool pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto quant_x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    const auto quant_x1 = weights_->quant_fc0.forward(quant_x0).apply_(relu<T>);
    const auto x1 = quant_x1.template mul_convert<T>(weights<T, Q>::fixed_dequantization_mul);
    const auto x2 = splice(x1, weights_->fc1.forward(x1).apply_(relu<T>));
    const auto x3 = splice(x2, weights_->fc2.forward(x2).apply_(relu<T>));
    return weights_->fc3.forward(x3).item();
  }

  inline search::score_type evaluate(const bool pov) const {
    const T eval = propagate(pov);
    const T value = search::logit_scale<T> * std::clamp(eval, search::min_logit<T>, search::max_logit<T>);
    return static_cast<search::score_type>(value);
  }

  eval(const weights<T, Q>* src) : weights_{src}, white{&src->quant_w}, black{&src->quant_b} {}
};

}  // namespace nnue
