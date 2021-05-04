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

#include <iostream>
#include <sstream>
#include <algorithm>
#include <numeric>
#include <string>
#include <cmath>
#include <cstdint>
#include <array>

#include <option_parser.h>
#include <move.h>

namespace search {

template <typename T>
inline constexpr T max_logit = static_cast<T>(256);

template <typename T>
inline constexpr T min_logit = static_cast<T>(-256);

template <typename T>
inline constexpr T logit_scale = static_cast<T>(1024);

template <typename T>
inline constexpr T wdl_scale = static_cast<T>(1024);

using depth_type = std::int32_t;

inline constexpr depth_type max_depth_ = 128;

inline constexpr depth_type max_depth_margin_ = 8;

using score_type = std::int32_t;

using wdl_type =
    std::tuple<search::score_type, search::score_type, search::score_type>;

inline constexpr score_type big_number = 256 * logit_scale<score_type>;

inline constexpr score_type max_mate_score = -2 * big_number;

inline constexpr score_type mate_score =
    max_mate_score - (max_depth_ + max_depth_margin_);

inline constexpr score_type draw_score = 0;

inline constexpr score_type aspiration_delta = 30;

inline constexpr score_type stability_threshold = 50;

using counter_type = std::int32_t;

using see_type = std::int32_t;

struct fixed_constants {
  static constexpr bool tuning = false;
  static constexpr depth_type lmr_tbl_dim = 64;
  size_t thread_count_;
  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{};

  const size_t& thread_count() const { return thread_count_; }

  constexpr depth_type reduce_depth() const { return 3; }
  constexpr depth_type max_depth() const { return max_depth_; }
  constexpr depth_type aspiration_depth() const { return 4; }
  constexpr depth_type nmp_depth() const { return 2; }
  constexpr depth_type history_prune_depth() const { return 8; }
  constexpr depth_type lmp_depth() const { return 7; }
  constexpr depth_type snmp_depth() const { return 7; }
  constexpr depth_type futility_prune_depth() const { return 6; }
  constexpr depth_type history_extension_depth() const { return 8; }

  constexpr depth_type reduction(const depth_type& depth,
                                 const int& move_idx) const {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim +
                   std::min(last_idx, move_idx)];
  }

  constexpr depth_type R(const depth_type& depth) const {
    return 4 + depth / 6;
  }

  constexpr counter_type history_prune_threshold(
      const bool& improving, const depth_type& depth) const {
    return static_cast<counter_type>(-256) * static_cast<counter_type>(depth) *
           static_cast<counter_type>(depth + improving);
  }

  constexpr depth_type history_extension_threshold() const {
    return static_cast<counter_type>(24576);
  }

  constexpr score_type futility_margin(const depth_type& depth) const {
    assert(depth > 0);
    constexpr score_type m = 2048;
    return m * static_cast<score_type>(depth);
  }

  constexpr score_type snmp_margin(const bool& improving,
                                   const depth_type& depth) const {
    assert(depth > 0);
    constexpr score_type m = 328;
    constexpr score_type b = 164;
    return m * static_cast<score_type>(depth - improving) + b;
  }

  constexpr size_t lmp_count(const bool& improving,
                             const depth_type& depth) const {
    constexpr std::array<size_t, 8> improving_counts = {0,  5,  8,  12,
                                                        20, 30, 42, 65};
    constexpr std::array<size_t, 8> worsening_counts = {0,  3,  4,  8,
                                                        10, 13, 21, 31};
    return improving ? improving_counts[depth] : worsening_counts[depth];
  }

  constexpr depth_type history_reduction(
      const counter_type& history_value) const {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / 5000);
    return std::max(-limit, std::min(limit, raw));
  }

  fixed_constants& update_(const size_t& thread_count) {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(
            0.75 + std::log(depth) * std::log(played) / 2.25);
      }
    }
    return *this;
  }

  auto options() { return engine::uci_options(); }

  fixed_constants(const size_t& thread_count = 1) { update_(thread_count); }
};

struct tuning_constants {
  static constexpr bool tuning = true;
  static constexpr depth_type lmr_tbl_dim = 64;
  size_t thread_count_;

  depth_type reduce_depth_{3};
  depth_type aspiration_depth_{4};
  depth_type nmp_depth_{2};
  depth_type history_prune_depth_{8};
  depth_type snmp_depth_{7};
  depth_type futility_prune_depth_{6};
  depth_type history_extension_depth_{8};

  depth_type R_bias_{4};
  depth_type R_div_{6};

  counter_type history_prune_threshold_mul_{-256};
  counter_type history_extension_threshold_{24576};

  score_type futility_margin_mul_{2048};
  score_type snmp_margin_mul_{328};
  score_type snmp_margin_bias_{164};

  counter_type history_reduction_div_{5000};
  double lmr_tbl_bias_{0.75};
  double lmr_tbl_div_{2.25};

  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{};

  const size_t& thread_count() const { return thread_count_; }
  constexpr depth_type max_depth() const { return max_depth_; }

  depth_type reduce_depth() const { return reduce_depth_; }
  depth_type aspiration_depth() const { return aspiration_depth_; }
  depth_type nmp_depth() const { return nmp_depth_; }
  depth_type history_prune_depth() const { return history_prune_depth_; }
  depth_type snmp_depth() const { return snmp_depth_; }
  depth_type futility_prune_depth() const { return futility_prune_depth_; }
  depth_type history_extension_depth() const {
    return history_extension_depth_;
  }
  constexpr depth_type lmp_depth() const { return 7; }

  depth_type reduction(const depth_type& depth, const int& move_idx) const {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim +
                   std::min(last_idx, move_idx)];
  }

  depth_type R(const depth_type& depth) const {
    return R_bias_ + depth / R_div_;
  }

  counter_type history_prune_threshold(const bool& improving,
                                       const depth_type& depth) const {
    return history_prune_threshold_mul_ * static_cast<counter_type>(depth) *
           static_cast<counter_type>(depth + improving);
  }

  depth_type history_extension_threshold() const {
    return history_extension_threshold_;
  }

  score_type futility_margin(const depth_type& depth) const {
    assert(depth > 0);
    return futility_margin_mul_ * static_cast<score_type>(depth);
  }

  constexpr score_type snmp_margin(const bool& improving,
                                   const depth_type& depth) const {
    assert(depth > 0);
    return snmp_margin_mul_ * static_cast<score_type>(depth - improving) +
           snmp_margin_bias_;
  }

  constexpr depth_type history_reduction(
      const counter_type& history_value) const {
    constexpr depth_type limit = 2;
    const depth_type raw =
        -static_cast<depth_type>(history_value / history_reduction_div_);
    return std::max(-limit, std::min(limit, raw));
  }

  constexpr size_t lmp_count(const bool& improving,
                             const depth_type& depth) const {
    constexpr std::array<size_t, 8> improving_counts = {0,  5,  8,  12,
                                                        20, 30, 42, 65};
    constexpr std::array<size_t, 8> worsening_counts = {0,  3,  4,  8,
                                                        10, 13, 21, 31};
    return improving ? improving_counts[depth] : worsening_counts[depth];
  }

  tuning_constants& update_(const size_t& thread_count) {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(
            lmr_tbl_bias_ + std::log(depth) * std::log(played) / lmr_tbl_div_);
      }
    }
    return *this;
  }

  auto options() {
    using namespace engine;
    auto option_reduce_depth = option_callback(
        spin_option("reduce_depth_", reduce_depth_, spin_range{2, 8}),
        [this](const int d) { reduce_depth_ = d; });

    auto option_aspiration_depth = option_callback(
        spin_option("aspiration_depth_", aspiration_depth_, spin_range{3, 8}),
        [this](const int d) { aspiration_depth_ = d; });

    auto option_nmp_depth =
        option_callback(spin_option("nmp_depth_", nmp_depth_, spin_range{2, 6}),
                        [this](const int d) { nmp_depth_ = d; });

    auto option_history_prune_depth =
        option_callback(spin_option("history_prune_depth_",
                                    history_prune_depth_, spin_range{4, 24}),
                        [this](const int d) { history_prune_depth_ = d; });

    auto option_snmp_depth = option_callback(
        spin_option("snmp_depth_", snmp_depth_, spin_range{2, 16}),
        [this](const int d) { snmp_depth_ = d; });

    auto option_futility_prune_depth =
        option_callback(spin_option("futility_prune_depth_",
                                    futility_prune_depth_, spin_range{2, 16}),
                        [this](const int d) { futility_prune_depth_ = d; });

    auto option_history_extension_depth = option_callback(
        spin_option("history_extension_depth_", history_extension_depth_,
                    spin_range{6, 16}),
        [this](const int d) { history_extension_depth_ = d; });

    auto option_R_bias =
        option_callback(spin_option("R_bias_", R_bias_, spin_range{1, 12}),
                        [this](const int d) { R_bias_ = d; });

    auto option_R_div =
        option_callback(spin_option("R_div_", R_div_, spin_range{1, 12}),
                        [this](const int d) { R_div_ = d; });

    auto option_history_prune_threshold_mul = option_callback(
        spin_option("history_prune_threshold_mul_",
                    history_prune_threshold_mul_, spin_range{-512, 0}),
        [this](const int d) { history_prune_threshold_mul_ = d; });

    auto option_history_extension_threshold = option_callback(
        spin_option("history_extension_threshold_",
                    history_extension_threshold_, spin_range{12288, 49152}),
        [this](const int d) { history_extension_threshold_ = d; });

    auto option_futility_margin_mul = option_callback(
        spin_option("futility_margin_mul_", futility_margin_mul_,
                    spin_range{512, 4096}),
        [this](const int d) { futility_margin_mul_ = d; });

    auto option_snmp_margin_mul =
        option_callback(spin_option("snmp_margin_mul_", snmp_margin_mul_,
                                    spin_range{128, 4096}),
                        [this](const int d) { snmp_margin_mul_ = d; });

    auto option_snmp_margin_bias = option_callback(
        spin_option("snmp_margin_bias_", snmp_margin_bias_, spin_range{0, 384}),
        [this](const int d) { snmp_margin_bias_ = d; });

    auto option_history_reduction_div = option_callback(
        spin_option("history_reduction_div_", history_reduction_div_,
                    spin_range{2048, 8192}),
        [this](const int d) { history_reduction_div_ = d; });

    auto option_lmr_tbl_bias = option_callback(
        string_option("lmr_tbl_bias_", std::to_string(lmr_tbl_bias_)),
        [this](const std::string d) {
          std::istringstream ss(d);
          ss >> lmr_tbl_bias_;
          update_(thread_count_);
        });

    auto option_lmr_tbl_div = option_callback(
        string_option("lmr_tbl_div_", std::to_string(lmr_tbl_div_)),
        [this](const std::string d) {
          std::istringstream ss(d);
          ss >> lmr_tbl_div_;
          update_(thread_count_);
        });

    return uci_options(
        option_reduce_depth, option_aspiration_depth, option_nmp_depth,
        option_history_prune_depth, option_snmp_depth,
        option_futility_prune_depth, option_history_extension_depth,
        option_R_bias, option_R_div, option_history_prune_threshold_mul,
        option_history_extension_threshold, option_futility_margin_mul,
        option_snmp_margin_mul, option_snmp_margin_bias,
        option_history_reduction_div, option_lmr_tbl_bias, option_lmr_tbl_div);
  }

  tuning_constants(const size_t& thread_count = 1) { update_(thread_count); }
};

using constants = fixed_constants;
}
