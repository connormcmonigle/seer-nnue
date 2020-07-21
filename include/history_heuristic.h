#pragma once

#include <iostream>
#include <array>
#include <type_traits>
#include <atomic>
#include <algorithm>

#include <enum_util.h>
#include <move.h>

namespace chess{

struct history_heuristic{
  static constexpr size_t num_squares = 64;
  static constexpr size_t num_pieces = 6;

  std::array<std::atomic_uint64_t, num_squares * num_squares * num_pieces> weights_{};

  size_t get_idx_(const move& mv) const {
    const size_t piece_idx = static_cast<size_t>(mv.piece());
    const size_t from_idx = static_cast<size_t>(mv.from().index());
    const size_t to_idx = static_cast<size_t>(mv.to().index());
    return num_squares * num_squares * piece_idx + num_squares * from_idx + to_idx;
  }

  history_heuristic& clear(){
    for(auto& elem : weights_){ elem = size_t{}; }
    return *this;
  }

  history_heuristic& add(const int& depth, const move& mv){
    if(!mv.is_capture()){
      const size_t idx = get_idx_(mv);
      weights_[idx] += static_cast<size_t>(depth) * static_cast<size_t>(depth);
    }
    return *this;
  }

  size_t count(const move& mv) const {
    return weights_[get_idx_(mv)].load(std::memory_order_relaxed);
  }

};

struct sided_history_heuristic : sided<sided_history_heuristic, history_heuristic> {
  history_heuristic white;
  history_heuristic black;
  sided_history_heuristic() : white{}, black{} {}
};

std::ostream& operator<<(std::ostream& ostr, const history_heuristic& hh){
  ostr << '{';
  for(const auto& val : hh.weights_){ ostr << val.load(std::memory_order_relaxed) << ", "; }
  return ostr << '}';
}

}