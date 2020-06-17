#pragma once

#include <cstdint>
#include <iostream>
#include <string>

#include <weights_streamer.h>
#include <nnue_util.h>

namespace nnue{

template<typename T>
struct half_kp{
  big_affine<T, 384*64, 256> w{};
  big_affine<T, 384*64, 256> b{};
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

  T propagate(bool pov) const {
    const auto w_x = stack_vector<T, 256>::from(w.incremental);
    const auto b_x = stack_vector<T, 256>::from(b.incremental);
    const auto x0 = pov ? splice(w_x, b_x).apply_(relu<T>) : splice(b_x, w_x).apply_(relu<T>);
    const auto x1 = fc0.forward(x0).apply_(relu<T>);
    const auto x2 = fc1.forward(x1).apply_(relu<T>);
    const T val = fc2.forward(x2).item();
    return val;
  }
  
  half_kp<T>& load(weights_streamer<T>& ws){
    w.load_(ws);
    b.load_(ws);
    fc0.load_(ws);
    fc1.load_(ws);
    fc2.load_(ws);
    return *this;
  }
  
  half_kp<T>& load(const std::string& path){
    auto ws = weights_streamer<T>(path);
    return load(ws);
  }
};

}
