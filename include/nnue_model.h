#pragma once

#include <cstdint>
#include <iostream>
#include <string>
#include <cmath>

#include <weights_streamer.h>
#include <nnue_util.h>
#include <enum_util.h>
#include <search_util.h>

namespace nnue{

constexpr size_t half_ka_numel = 768*64;
constexpr size_t base_dim = 128;

template<typename T>
struct weights{
  typename weights_streamer<T>::signature_type signature_{0};
  big_affine<T, half_ka_numel, base_dim> w{};
  big_affine<T, half_ka_numel, base_dim> b{};
  stack_affine<T, 2*base_dim, 32> fc0{};
  stack_affine<T, 32, 32> fc1{};
  stack_affine<T, 64, 32> fc2{};
  stack_affine<T, 96, 3> fc3{};

  size_t signature() const { return signature_; }
  
  size_t num_parameters() const {
    return w.num_parameters() +
           b.num_parameters() +
           fc0.num_parameters() +
           fc1.num_parameters() +
           fc2.num_parameters() +
           fc3.num_parameters();
  }
  
  weights<T>& load(weights_streamer<T>& ws){
    w.load_(ws);
    b.load_(ws);
    fc0.load_(ws);
    fc1.load_(ws);
    fc2.load_(ws);
    fc3.load_(ws);
    signature_ = ws.signature();
    return *this;
  }
  
  weights<T>& load(const std::string& path){
    auto ws = weights_streamer<T>(path);
    return load(ws);
  }
};

template<typename T>
struct feature_transformer{
  const big_affine<T, half_ka_numel, base_dim>* weights_;
  stack_vector<T, base_dim> active_;
  constexpr stack_vector<T, base_dim> active() const { return active_; }

  void clear(){ active_ = stack_vector<T, base_dim>::from(weights_ -> b); }
  void insert(const size_t idx){ weights_ -> insert_idx(idx, active_); }
  void erase(const size_t idx){ weights_ -> erase_idx(idx, active_); }

  feature_transformer(const big_affine<T, half_ka_numel, base_dim>* src) : weights_{src} {
    clear();
  }
};

template<typename T>
struct eval : chess::sided<eval<T>, feature_transformer<T>>{
  const weights<T>* weights_;
  feature_transformer<T> white;
  feature_transformer<T> black;

  constexpr stack_vector<T, 3> propagate(const bool pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    const auto x1 = (weights_ -> fc0).forward(x0).apply_(relu<T>);
    const auto x2 = splice(x1, (weights_ -> fc1).forward(x1).apply_(relu<T>));
    const auto x3 = splice(x2, (weights_ -> fc2).forward(x2).apply_(relu<T>));
    return (weights_ -> fc3).forward(x3).softmax_();
  }

  constexpr search::wdl_type wdl(const bool pov) const {
    auto map = [](const T x){ return  static_cast<search::score_type>(search::wdl_scale<T> * x); };
    const stack_vector<T, 3> wdl = propagate(pov);
    return std::tuple(map(wdl.data[0]), map(wdl.data[1]), map(wdl.data[2]));
  }

  constexpr T evaluate(const bool pov) const {
    constexpr T one = static_cast<T>(1.0);
    constexpr T half = static_cast<T>(0.5);
    constexpr T epsilon = static_cast<T>(0.0001);

    const stack_vector<T, 3> wdl = propagate(pov);
    const T advantage = (wdl.data[0] + half * wdl.data[1]) / (wdl.data[2] + half * wdl.data[1]);
    const T eval = std::log(std::clamp(advantage, epsilon, one-epsilon));

    const T value = search::logit_scale<T> * std::clamp(eval, search::min_logit<T>, search::max_logit<T>);
    return static_cast<search::score_type>(value);
  }
  
  eval(const weights<T>* src) : weights_{src}, white{&(src -> w)}, black{&(src -> b)} {}
};

}
