#pragma once

#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <cstdint>
#include <array>


#include <move.h>

namespace search{

template<typename T>
inline constexpr T max_logit = static_cast<T>(256);

template<typename T>
inline constexpr T min_logit = static_cast<T>(-256);

template<typename T>
inline constexpr T logit_scale = static_cast<T>(1024);

using depth_type = std::int32_t;

inline constexpr depth_type max_depth_ = 128;

inline constexpr depth_type max_depth_margin_ = 8;

using score_type = std::int32_t;

inline constexpr score_type big_number = 256 * logit_scale<score_type>;

inline constexpr score_type max_mate_score = -2 * big_number;

inline constexpr score_type mate_score = max_mate_score - (max_depth_ + max_depth_margin_);

inline constexpr score_type draw_score = 0;

inline constexpr score_type aspiration_delta = 30;

inline constexpr score_type stability_threshold = 50;
  

using counter_type = std::int32_t;

using see_type = std::int32_t;

struct constants{
  static constexpr depth_type lmr_tbl_dim = 64;
  size_t thread_count_;
  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{}; 
  
  const size_t& thread_count() const { return thread_count_; }
  constexpr depth_type reduce_depth() const { return 3; }
  constexpr depth_type max_depth() const { return max_depth_; }
  constexpr depth_type aspiration_depth() const { return 4; }
  constexpr depth_type nmp_depth() const { return 2; }
  constexpr depth_type history_prune_depth() const { return 8; }
  constexpr depth_type snmp_depth() const { return 7; }
  constexpr depth_type futility_prune_depth() const { return 6; }
  constexpr depth_type history_extension_depth() const { return 8; }
  
  constexpr depth_type reduction(const depth_type& depth, const int& move_idx) const {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }
  
  constexpr depth_type R(const depth_type& depth) const {
    return 4 + depth / 6;
  }

  constexpr counter_type history_prune_threshold(const bool& improving, const depth_type& depth) const {
    return  static_cast<counter_type>(-256) * static_cast<counter_type>(depth) * static_cast<counter_type>(depth + improving);
  }

  constexpr depth_type history_extension_threshold() const { return static_cast<counter_type>(24576); }

  
  constexpr score_type futility_margin(const depth_type& depth) const {
    assert(depth > 0);
    constexpr score_type m = 2048;
    return m * static_cast<score_type>(depth);
  }
  
  constexpr score_type snmp_margin(const bool& improving, const depth_type& depth) const {
    assert(depth > 0);
    constexpr score_type m = 328;
    constexpr score_type b = 164;
    return m * static_cast<score_type>(depth - improving) + b;
  }

  constexpr depth_type history_reduction(const counter_type& history_value) const {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / 5000);
    return std::max(-limit, std::min(limit, raw));
  }

  constants& update_(const size_t& thread_count){
    thread_count_ = thread_count;
    for(depth_type depth{1}; depth < lmr_tbl_dim; ++depth){
      for(depth_type played{1}; played < lmr_tbl_dim; ++played){
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(0.75 + std::log(depth) * std::log(played) / 2.25);
      }
    }
    return *this;
  }
  
  constants(const size_t& thread_count){ update_(thread_count); }
};

}
