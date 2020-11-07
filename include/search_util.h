#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <array>


#include <move.h>

namespace search{

using depth_type = int;
inline constexpr depth_type max_depth_ = 128;

struct constants{
  static constexpr search::depth_type lmr_tbl_dim = 64;
  size_t thread_count_;
  std::array<search::depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{}; 
  
  const size_t& thread_count() const { return thread_count_; }
  constexpr depth_type reduce_depth() const { return 3; }
  constexpr depth_type max_depth() const { return max_depth_; }
  constexpr depth_type aspiration_depth() const { return 4; }
  constexpr depth_type nmp_depth() const { return 2; }
  constexpr depth_type history_prune_depth() const { return 2; }
  constexpr depth_type snmp_depth() const { return 7; }
  constexpr depth_type futility_prune_depth() const { return 6; }

  
  template<bool is_pv>
  constexpr depth_type reduction(const depth_type& depth, const int& move_idx) const {
    constexpr search::depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }
  
  constexpr depth_type R(const depth_type& depth) const {
    return 4 + depth / 6;
  }

  template<typename H>
  constexpr H history_prune_threshold() const { return static_cast<H>(0); }

  template<typename T>
  constexpr T futility_margin(const depth_type& depth) const {
    assert(depth > 0);
    constexpr T m = static_cast<T>(1.5);
    return m * static_cast<T>(depth);
  }
  
  template<typename T>
  constexpr T snmp_margin(const bool& improving, const depth_type& depth) const {
    assert(depth > 0);
    constexpr T m = static_cast<T>(0.32);
    constexpr T b = static_cast<T>(0.16);
    return m * static_cast<T>(depth - improving) + b;
  }

  template<typename H>
  constexpr depth_type history_reduction(H& history_value) const {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / 5000);
    return std::max(-limit, std::min(limit, raw));
  }

  constants& update_(const size_t& thread_count){
    thread_count_ = thread_count;
    for(search::depth_type depth{1}; depth < lmr_tbl_dim; ++depth){
      for(search::depth_type played{1}; played < lmr_tbl_dim; ++played){
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<search::depth_type>(0.75 + std::log(depth) * std::log(played) / 2.25);
      }
    }
    return *this;
  }
  
  constants(const size_t& thread_count){ update_(thread_count); }
};



}
