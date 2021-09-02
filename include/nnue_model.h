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

constexpr size_t half_ka_numel = 64 * 12 * 64;
constexpr size_t half_pawn_numel = 64 * 2 * 64;
constexpr size_t max_active_half_features = 32;
constexpr size_t max_active_half_pawn_features = 16;

constexpr size_t base_dim = 160;
constexpr size_t p_base_dim = 512;

template <typename T>
struct weights {
  using p_encoding_type = stack_vector<T, 16>;

  typename weights_streamer<T>::signature_type signature_{0};
  big_affine<T, half_ka_numel, base_dim> w{};
  big_affine<T, half_ka_numel, base_dim> b{};
  big_affine<T, half_pawn_numel, p_base_dim> p_w{};
  big_affine<T, half_pawn_numel, p_base_dim> p_b{};
  stack_affine<T, 2 * base_dim, 16> fc0{};
  stack_affine<T, 2 * p_base_dim, 16> p_fc0{};
  stack_affine<T, 16, 16> fc1{};
  stack_affine<T, 32, 16> fc2{};
  stack_affine<T, 48, 1> fc3{};

  size_t signature() const { return signature_; }

  size_t num_parameters() const {
    return w.num_parameters() + b.num_parameters() + p_w.num_parameters() + p_b.num_parameters() + fc0.num_parameters() + p_fc0.num_parameters() +
           fc1.num_parameters() + fc2.num_parameters() + fc3.num_parameters();
  }

  template <typename streamer_type>
  weights<T>& load(streamer_type& ws) {
    w.load_(ws);
    b.load_(ws);
    p_w.load_(ws);
    p_b.load_(ws);
    fc0.load_(ws);
    p_fc0.load_(ws);
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

template <typename T, size_t feature_dim, size_t output_dim>
struct feature_transformer {
  const big_affine<T, feature_dim, output_dim>* weights_;
  stack_vector<T, output_dim> active_;
  constexpr stack_vector<T, output_dim> active() const { return active_; }

  void clear() { active_ = stack_vector<T, output_dim>::from(weights_->b); }
  void insert(const size_t idx) { weights_->insert_idx(idx, active_); }
  void erase(const size_t idx) { weights_->erase_idx(idx, active_); }

  feature_transformer(const big_affine<T, feature_dim, output_dim>* src) : weights_{src} { clear(); }
};

template <typename T>
struct p_eval : chess::sided<p_eval<T>, feature_transformer<T, half_pawn_numel, p_base_dim>> {
  using p_encoding_type = typename weights<T>::p_encoding_type;

  const weights<T>* weights_;
  feature_transformer<T, half_pawn_numel, p_base_dim> white;
  feature_transformer<T, half_pawn_numel, p_base_dim> black;

  inline p_encoding_type propagate(const bool& pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    return weights_->p_fc0.forward(x0);
  }

  p_eval(const weights<T>* src) : weights_{src}, white{&src->p_w}, black{&src->p_b} {}
};

template <typename T>
struct eval : chess::sided<eval<T>, feature_transformer<T, half_ka_numel, base_dim>> {
  using p_encoding_type = typename weights<T>::p_encoding_type;

  const weights<T>* weights_;
  feature_transformer<T, half_ka_numel, base_dim> white;
  feature_transformer<T, half_ka_numel, base_dim> black;

  inline T propagate(const bool& pov, const p_encoding_type& p_encoding) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    const auto x1 = weights_->fc0.forward(x0).add_(p_encoding.data).apply_(relu<T>);
    const auto x2 = splice(x1, weights_->fc1.forward(x1).apply_(relu<T>));
    const auto x3 = splice(x2, weights_->fc2.forward(x2).apply_(relu<T>));
    return weights_->fc3.forward(x3).item();
  }

  inline search::score_type evaluate(const bool& pov, const typename weights<T>::p_encoding_type& p_encoding) const {
    const T eval = propagate(pov, p_encoding);
    const T value = search::logit_scale<T> * std::clamp(eval, search::min_logit<T>, search::max_logit<T>);
    return static_cast<search::score_type>(value);
  }

  eval(const weights<T>* src) : weights_{src}, white{&src->w}, black{&src->b} {}
};

}  // namespace nnue
