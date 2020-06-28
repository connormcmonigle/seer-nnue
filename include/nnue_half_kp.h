#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include <weights_streamer.h>
#include <nnue_util.h>
#include <enum_util.h>

namespace nnue{

template<typename T>
struct half_kp_weights{
  big_affine<T, 768*64, 256> w{};
  big_affine<T, 768*64, 256> b{};
  stack_affine<T, 512, 32> fc0{};
  stack_affine<T, 32, 32> fc1{};
  stack_affine<T, 32, 1> fc2{};

  size_t num_parameters() const {
    return w.num_parameters() +
           b.num_parameters() +
           fc0.num_parameters() +
           fc1.num_parameters() +
           fc2.num_parameters();
  }
  
  half_kp_weights<T>& load(weights_streamer<T>& ws){
    w.load_(ws);
    b.load_(ws);
    fc0.load_(ws);
    fc1.load_(ws);
    fc2.load_(ws);
    return *this;
  }
  
  half_kp_weights<T>& load(const std::string& path){
    auto ws = weights_streamer<T>(path);
    return load(ws);
  }
};

template<typename T>
struct feature_transformer{
  const big_affine<T, 768*64, 256>* weights_;
  stack_vector<T, 256> active_;
  constexpr stack_vector<T, 256> active() const { return active_; }

  void clear(){ active_ = stack_vector<T, 256>::from(weights_ -> b); }
  void insert(const size_t idx){ weights_ -> insert_idx(idx, active_); }
  void erase(const size_t idx){ weights_ -> erase_idx(idx, active_); }

  feature_transformer(const big_affine<T, 768*64, 256>* src) : weights_{src} {
    clear();
  }
};

template<typename T>
struct half_kp_eval : chess::sided<half_kp_eval<T>, feature_transformer<T>>{
  const half_kp_weights<T>* weights_;
  feature_transformer<T> white;
  feature_transformer<T> black;

  constexpr T propagate(bool pov) const {
    const auto w_x = white.active();
    const auto b_x = black.active();
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    const auto x1 = (weights_ -> fc0).forward(x0).apply_(relu<T>);
    const auto x2 = (weights_ -> fc1).forward(x1).apply_(relu<T>);
    const T val = (weights_ -> fc2).forward(x2).item();
    return val;
  }

  half_kp_eval(const half_kp_weights<T>* src) : weights_{src}, white{&(src -> w)}, black{&(src -> b)} {}
};

}
