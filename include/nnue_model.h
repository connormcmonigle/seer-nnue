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
constexpr size_t base_dim = 256;

struct weights {
  using parameter_type = float;

  weights_streamer::signature_type signature_{0};
  big_affine<parameter_type, half_ka_numel, base_dim> w{};
  big_affine<parameter_type, half_ka_numel, base_dim> b{};
  stack_affine<parameter_type, 2 * base_dim, 1> fc0{};

  size_t signature() const { return signature_; }

  size_t num_parameters() const { return w.num_parameters() + b.num_parameters() + fc0.num_parameters(); }

  template <typename streamer_type>
  weights& load(streamer_type& ws) {
    w.load_(ws);
    b.load_(ws);
    fc0.load_(ws);
    signature_ = ws.signature();
    return *this;
  }

  weights& load(const std::string& path) {
    auto ws = weights_streamer(path);
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

struct eval : chess::sided<eval, feature_transformer<weights::parameter_type>> {
  using parameter_type = weights::parameter_type;

  const weights* weights_;
  feature_transformer<parameter_type> white;
  feature_transformer<parameter_type> black;

  inline parameter_type propagate(const bool pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<parameter_type>) : splice(b_x, w_x).apply_(relu<parameter_type>);
    return weights_->fc0.forward(x0).item();
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

  eval(const weights* src) : weights_{src}, white{&src->w}, black{&src->b} {}
};

}  // namespace nnue
