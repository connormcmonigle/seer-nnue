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

#include <move.h>
#include <option_parser.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>

namespace search {

template <typename T>
inline constexpr T max_logit = static_cast<T>(8);

template <typename T>
inline constexpr T min_logit = static_cast<T>(-8);

template <typename T>
inline constexpr T logit_scale = static_cast<T>(1024);

template <typename T>
inline constexpr T wdl_scale = static_cast<T>(1024);

using depth_type = std::int32_t;

inline constexpr depth_type max_depth = 128;

inline constexpr depth_type max_depth_margin = 8;

using score_type = std::int32_t;

using wdl_type = std::tuple<search::score_type, search::score_type, search::score_type>;

inline constexpr score_type big_number = 8 * logit_scale<score_type>;

inline constexpr score_type max_mate_score = -2 * big_number;

inline constexpr score_type mate_score = max_mate_score - (max_depth + max_depth_margin);

inline constexpr score_type draw_score = 0;

inline constexpr score_type aspiration_delta = 20;

inline constexpr score_type stability_threshold = 50;

using counter_type = std::int32_t;

using see_type = std::int32_t;

inline constexpr size_t nodes_per_update = 512;

struct fixed_constants {
  static constexpr bool tuning = false;
  static constexpr depth_type lmr_tbl_dim = 64;
  size_t thread_count_;
  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{};

  const size_t& thread_count() const { return thread_count_; }

  constexpr depth_type reduce_depth() const { return 3; }
  constexpr depth_type aspiration_depth() const { return 4; }
  constexpr depth_type nmp_depth() const { return 2; }
  constexpr depth_type lmp_depth() const { return 7; }
  constexpr depth_type snmp_depth() const { return 7; }
  constexpr depth_type futility_prune_depth() const { return 6; }
  constexpr depth_type quiet_see_prune_depth() const { return 8; }
  constexpr depth_type noisy_see_prune_depth() const { return 6; }
  constexpr depth_type history_extension_depth() const { return 8; }
  constexpr depth_type singular_extension_depth() const { return 9; }
  constexpr depth_type iir_depth() const { return 4; }
  constexpr depth_type prob_prune_depth() const { return 4; }

  constexpr depth_type reduction(const depth_type& depth, const int& move_idx) const {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }

  constexpr depth_type nmp_reduction(const depth_type& depth, const score_type& beta, const score_type& value) const { return 4 + depth / 6 + std::min(3, (value - beta) / 256); }

  constexpr see_type nmp_see_threshold() const { return 200; }

  constexpr depth_type history_extension_threshold() const { return 24576; }

  constexpr depth_type singular_extension_depth_margin() const { return 2; }

  constexpr depth_type singular_search_depth(const depth_type& depth) const { return depth / 2 - 1; }

  constexpr score_type singular_beta(const score_type& tt_score, const depth_type& depth) const {
    return tt_score - 2 * static_cast<score_type>(depth);
  }

  constexpr score_type singular_double_extension_margin() const { return 160; }

  constexpr depth_type prob_prune_depth_margin() const { return 2; }

  constexpr score_type prob_prune_margin() const { return 768; }

  constexpr score_type futility_margin(const depth_type& depth) const {
    assert(depth > 0);
    constexpr score_type m = 1536;
    return m * static_cast<score_type>(depth);
  }

  constexpr score_type snmp_margin(const bool& improving, const depth_type& depth) const {
    assert(depth > 0);
    constexpr score_type m = 328;
    constexpr score_type b = 164;
    return m * static_cast<score_type>(depth - improving) + b;
  }

  constexpr int lmp_count(const bool& improving, const depth_type& depth) const {
    constexpr std::array<int, 8> improving_counts = {0, 5, 8, 12, 20, 30, 42, 65};
    constexpr std::array<int, 8> worsening_counts = {0, 3, 4, 8, 10, 13, 21, 31};
    return improving ? improving_counts[depth] : worsening_counts[depth];
  }

  constexpr see_type quiet_see_prune_threshold(const depth_type& depth) const { return -50 * static_cast<see_type>(depth); }
  constexpr see_type noisy_see_prune_threshold(const depth_type& depth) const { return -30 * static_cast<see_type>(depth * depth); }

  constexpr counter_type history_prune_threshold(const depth_type& depth) const { return -1024 * static_cast<counter_type>(depth * depth); }

  constexpr depth_type history_reduction(const counter_type& history_value) const {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / 5000);
    return std::clamp(raw, -limit, limit);
  }

  constexpr score_type delta_margin() const {
    constexpr score_type margin = 512;
    return margin;
  }

  fixed_constants& update_(const size_t& thread_count) {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(0.75 + std::log(depth) * std::log(played) / 2.25);
      }
    }
    return *this;
  }

  auto options() { return engine::uci_options(); }

  fixed_constants(const size_t& thread_count = 1) { update_(thread_count); }
};

struct tuning_constants : fixed_constants {
  static constexpr bool tuning = true;
  static constexpr depth_type lmr_tbl_dim = 64;

  depth_type reduce_depth_{3};
  double lmr_tbl_bias_{0.75};
  double lmr_tbl_div_{2.25};

  depth_type reduce_depth() const { return reduce_depth_; }

  depth_type reduction(const depth_type& depth, const int& move_idx) const {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }

  tuning_constants& update_(const size_t& thread_count) {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(lmr_tbl_bias_ + std::log(depth) * std::log(played) / lmr_tbl_div_);
      }
    }
    return *this;
  }

  auto options() {
    using namespace engine;
    auto option_reduce_depth =
        option_callback(spin_option("reduce_depth_", reduce_depth_, spin_range{2, 8}), [this](const int d) { reduce_depth_ = d; });

    auto option_lmr_tbl_bias = option_callback(string_option("lmr_tbl_bias_", std::to_string(lmr_tbl_bias_)), [this](const std::string d) {
      std::istringstream ss(d);
      ss >> lmr_tbl_bias_;
      update_(thread_count_);
    });

    auto option_lmr_tbl_div = option_callback(string_option("lmr_tbl_div_", std::to_string(lmr_tbl_div_)), [this](const std::string d) {
      std::istringstream ss(d);
      ss >> lmr_tbl_div_;
      update_(thread_count_);
    });

    return uci_options(option_reduce_depth, option_lmr_tbl_bias, option_lmr_tbl_div);
  }

  tuning_constants(const size_t& thread_count = 1) { update_(thread_count); }
};

#ifdef TUNE
using constants = tuning_constants;
#else
using constants = fixed_constants;
#endif

}  // namespace search
