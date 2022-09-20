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

  static constexpr parameter_type shared_quantization_scale = static_cast<parameter_type>(512);
  static constexpr parameter_type fc0_weight_quantization_scale = static_cast<parameter_type>(1024);
  static constexpr parameter_type fc0_bias_quantization_scale = shared_quantization_scale * fc0_weight_quantization_scale;
  static constexpr parameter_type dequantization_scale = static_cast<parameter_type>(1) / (shared_quantization_scale * fc0_weight_quantization_scale);

  static constexpr size_t base_dim = 512;

  weights_streamer::signature_type signature_{0};
  big_affine<parameter_type, feature::half_ka::numel, base_dim> shared{};
  big_affine<quantized_parameter_type, feature::half_ka::numel, base_dim> quantized_shared{};

  stack_affine<parameter_type, 2 * base_dim, 8> fc0{};
  stack_affine<quantized_parameter_type, 2 * base_dim, 8> quantized_fc0{};

  stack_affine<parameter_type, 8, 8> fc1{};
  stack_affine<parameter_type, 16, 8> fc2{};
  stack_affine<parameter_type, 24, 1> fc3{};

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
    quantized_fc0 = fc0.quantized<quantized_parameter_type>(fc0_weight_quantization_scale, fc0_bias_quantization_scale);

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
  stack_vector<T, weights::base_dim> active_;
  constexpr stack_vector<T, weights::base_dim> active() const { return active_; }

  void clear() { active_.set_(weights_->b); }
  void insert(const size_t idx) { weights_->insert_idx(idx, active_); }
  void erase(const size_t idx) { weights_->erase_idx(idx, active_); }

  feature_transformer(const big_affine<T, feature::half_ka::numel, weights::base_dim>* src) : weights_{src} { clear(); }
};

struct eval : chess::sided<eval, feature_transformer<weights::quantized_parameter_type>> {
  using parameter_type = weights::parameter_type;
  using quantized_parameter_type = weights::quantized_parameter_type;

  const weights* weights_;
  feature_transformer<quantized_parameter_type> white;
  feature_transformer<quantized_parameter_type> black;

  inline parameter_type propagate(const bool pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<quantized_parameter_type>) : splice(b_x, w_x).apply_(relu<quantized_parameter_type>);
    const auto x1 = weights_->quantized_fc0.forward(x0).dequantized<parameter_type>(weights::dequantization_scale).apply_(relu<parameter_type>);
    const auto x2 = splice(x1, weights_->fc1.forward(x1).apply_(relu<parameter_type>));
    const auto x3 = splice(x2, weights_->fc2.forward(x2).apply_(relu<parameter_type>));
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

  eval(const weights* src) : weights_{src}, white{&src->quantized_shared}, black{&src->quantized_shared} {}
};

struct eval_node {
  struct context {
    eval_node* parent_node_{nullptr};
    const chess::board* parent_board_{nullptr};
    const chess::move move_{chess::move::null()};
  };

  bool dirty_;

  union {
    context context_;
    eval eval_;
  } data_;

  bool dirty() const { return dirty_; }

  const eval& evaluator() {
    if (!dirty_) { return data_.eval_; }
    dirty_ = false;
    const context ctxt = data_.context_;
    data_.eval_ = ctxt.parent_board_->apply_update(ctxt.move_, ctxt.parent_node_->evaluator());
    return data_.eval_;
  }

  eval_node dirty_child(const chess::board* bd, const chess::move& mv) { return eval_node::dirty_node(context{this, bd, mv}); }

  static eval_node dirty_node(const context& context) { return eval_node{true, {context}}; }

  static eval_node clean_node(const eval& eval) {
    eval_node result{};
    result.dirty_ = false;
    result.data_.eval_ = eval;
    return result;
  }
};

}  // namespace nnue
