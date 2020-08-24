#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

#include <move.h>

namespace search{

using depth_type = int;

struct constants{
  size_t thread_count_;
  
  const size_t& thread_count() const { return thread_count_; }
  constexpr depth_type reduce_depth() const { return 3; }
  constexpr depth_type max_depth() const { return 128; }
  constexpr depth_type aspiration_depth() const { return 4; }
  constexpr depth_type nmp_depth() const { return 2; }
  
  template<bool is_pv>
  constexpr depth_type reduction(const depth_type& depth, const size_t& move_idx) const {
    if(move_idx == 0) return 0;
    if(move_idx < 6) return 1;
    if constexpr(is_pv){
      return depth / 4;
    }else{
      return depth / 2;
    }
  }
  
  constexpr depth_type R(const depth_type& depth){
    return 4 + depth / 6;
  }

  constants& update_(const size_t& thread_count){
    thread_count_ = thread_count;
    return *this;
  }

  constants(const size_t& thread_count){ update_(thread_count); }
};



}
