/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

#include <engine/option_parser.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>

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

inline constexpr depth_type reduction_offset_scale = 1024;

constexpr depth_type safe_depth = max_depth + max_depth_margin;

using score_type = std::int32_t;

using wdl_type = std::tuple<score_type, score_type, score_type>;

inline constexpr score_type big_number = 8 * logit_scale<score_type>;

inline constexpr score_type max_mate_score = -2 * big_number;

inline constexpr score_type mate_score = max_mate_score - (max_depth + max_depth_margin);

inline constexpr score_type tb_win_score = big_number + 1;

inline constexpr score_type tb_loss_score = -tb_win_score;

inline constexpr score_type draw_score = 0;

using counter_type = std::int32_t;

using see_type = std::int32_t;

inline constexpr std::size_t nodes_per_update = 512;

struct fixed_search_constants {
  static constexpr bool tuning = false;
  static constexpr depth_type lmr_tbl_dim = 64;
  std::size_t thread_count_;
  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{};

  [[nodiscard]] const fixed_search_constants& fixed() const noexcept { return *this; }
  [[nodiscard]] const std::size_t& thread_count() const noexcept { return thread_count_; }

  [[nodiscard]] constexpr depth_type reduce_depth() const noexcept { return 2; }
  [[nodiscard]] constexpr depth_type aspiration_depth() const noexcept { return 4; }
  [[nodiscard]] constexpr depth_type nmp_depth() const noexcept { return 3; }
  [[nodiscard]] constexpr depth_type lmp_depth() const noexcept { return 7; }
  [[nodiscard]] constexpr depth_type snmp_depth() const noexcept { return 6; }
  [[nodiscard]] constexpr depth_type futility_prune_depth() const noexcept { return 5; }
  [[nodiscard]] constexpr depth_type quiet_see_prune_depth() const noexcept { return 9; }
  [[nodiscard]] constexpr depth_type noisy_see_prune_depth() const noexcept { return 6; }
  [[nodiscard]] constexpr depth_type singular_extension_depth() const noexcept { return 6; }
  [[nodiscard]] constexpr depth_type probcut_depth() const noexcept { return 5; }
  [[nodiscard]] constexpr depth_type iir_depth() const noexcept { return 2; }

  [[nodiscard]] constexpr depth_type aspiration_delta() const noexcept { return 21; }

  [[nodiscard]] constexpr depth_type reduction(const depth_type& depth, const int& move_idx) const noexcept {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }

  [[nodiscard]] constexpr depth_type nmp_reduction(const depth_type& depth, const score_type& beta, const score_type& value) const noexcept {
    return 3 + depth / 4 + std::min(4, (value - beta) / 242);
  }

  [[nodiscard]] constexpr see_type nmp_see_threshold() const noexcept { return 226; }

  [[nodiscard]] constexpr depth_type singular_extension_depth_margin() const noexcept { return 3; }

  [[nodiscard]] constexpr depth_type singular_search_depth(const depth_type& depth) const noexcept { return depth / 2 - 1; }

  [[nodiscard]] constexpr score_type singular_beta(const score_type& tt_score, const depth_type& depth) const noexcept {
    return tt_score - 2 * static_cast<score_type>(depth);
  }

  [[nodiscard]] constexpr score_type singular_double_extension_margin() const noexcept { return 166; }

  [[nodiscard]] constexpr score_type futility_margin(const depth_type& depth) const noexcept {
    constexpr score_type m = 1544;
    return m * static_cast<score_type>(depth);
  }

  [[nodiscard]] constexpr score_type snmp_margin(const bool& improving, const bool& threats, const depth_type& depth) const noexcept {
    constexpr score_type m = 297;
    constexpr score_type b = 112;
    return m * static_cast<score_type>(depth - (improving && !threats)) + (threats ? b : 0);
  }

  [[nodiscard]] constexpr int lmp_count(const bool& improving, const depth_type& depth) const noexcept {
    constexpr std::array<int, 8> improving_counts = {0, 5, 8, 12, 20, 30, 42, 65};
    constexpr std::array<int, 8> worsening_counts = {0, 3, 4, 8, 10, 13, 21, 31};
    return improving ? improving_counts[depth] : worsening_counts[depth];
  }

  [[nodiscard]] constexpr see_type quiet_see_prune_threshold(const depth_type& depth) const noexcept { return -54 * static_cast<see_type>(depth); }
  [[nodiscard]] constexpr see_type noisy_see_prune_threshold(const depth_type& depth) const noexcept { return -111 * static_cast<see_type>(depth); }

  [[nodiscard]] constexpr counter_type history_prune_threshold(const depth_type& depth) const noexcept {
    return -1397 * static_cast<counter_type>(depth * depth);
  }

  [[nodiscard]] constexpr depth_type history_reduction(const counter_type& history_value) const noexcept {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / 5872);
    return std::clamp(raw, -limit, limit);
  }

  [[nodiscard]] constexpr score_type delta_margin() const noexcept {
    constexpr score_type margin = 506;
    return margin;
  }

  [[nodiscard]] constexpr see_type good_capture_prune_see_margin() const noexcept { return 270; }
  [[nodiscard]] constexpr score_type good_capture_prune_score_margin() const noexcept { return 265; }

  [[nodiscard]] constexpr depth_type probcut_search_depth(const depth_type& depth) const noexcept { return depth - 3; }
  [[nodiscard]] constexpr score_type probcut_beta(const score_type& beta) const noexcept { return beta + 315; }

  [[nodiscard]] constexpr depth_type razor_depth() const noexcept { return 3; }
  [[nodiscard]] constexpr score_type razor_margin(const depth_type& depth) const noexcept { return 896 * depth; }

  [[maybe_unused]] fixed_search_constants& update_(const std::size_t& thread_count) noexcept {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(0.4711025619218822 + std::log(depth) * std::log(played) / 2.5650906222921046);
      }
    }
    return *this;
  }

  auto options() noexcept { return engine::uci_options<>(); }

  explicit fixed_search_constants(const std::size_t& thread_count = 1) noexcept { update_(thread_count); }
};

#define INTEGRAL_OPTION(VALUE, A, B, C_END, R_END) tuning_option_<int>(#VALUE, VALUE, (A), (B), (C_END), (R_END))
#define FLOATING_OPTION(VALUE, A, B, C_END, R_END) tuning_option_<double>(#VALUE, VALUE, (A), (B), (C_END), (R_END))

struct tuning_search_constants : fixed_search_constants {
  static constexpr bool tuning = true;
  static constexpr depth_type lmr_tbl_dim = 64;

  template <typename T>
  auto tuning_option_(std::string name, T& value, T a, T b, double c_end, double r_end) {
    const auto option = engine::tune_option<T>(name, value, engine::value_range(a, b)).set_c_end(c_end).set_r_end(r_end);
    return engine::option_callback(option, [this, &value](const T& x) {
      value = x;
      update_(thread_count_);
    });
  }

  std::size_t thread_count_;
  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{};

  depth_type reduce_depth_{fixed().reduce_depth()};
  depth_type aspiration_depth_{fixed().aspiration_depth()};

  depth_type nmp_depth_{fixed().nmp_depth()};
  depth_type snmp_depth_{fixed().snmp_depth()};
  depth_type futility_prune_depth_{fixed().futility_prune_depth()};

  depth_type quiet_see_prune_depth_{fixed().quiet_see_prune_depth()};
  depth_type noisy_see_prune_depth_{fixed().noisy_see_prune_depth()};

  depth_type lmp_worsening_1_{fixed().lmp_count(false, 1)};
  depth_type lmp_worsening_2_{fixed().lmp_count(false, 2)};
  depth_type lmp_worsening_3_{fixed().lmp_count(false, 3)};
  depth_type lmp_worsening_4_{fixed().lmp_count(false, 4)};
  depth_type lmp_worsening_5_{fixed().lmp_count(false, 5)};
  depth_type lmp_worsening_6_{fixed().lmp_count(false, 6)};
  depth_type lmp_worsening_7_{fixed().lmp_count(false, 7)};

  depth_type lmp_improving_1_{fixed().lmp_count(true, 1)};
  depth_type lmp_improving_2_{fixed().lmp_count(true, 2)};
  depth_type lmp_improving_3_{fixed().lmp_count(true, 3)};
  depth_type lmp_improving_4_{fixed().lmp_count(true, 4)};
  depth_type lmp_improving_5_{fixed().lmp_count(true, 5)};
  depth_type lmp_improving_6_{fixed().lmp_count(true, 6)};
  depth_type lmp_improving_7_{fixed().lmp_count(true, 7)};

  depth_type singular_extension_depth_{fixed().singular_extension_depth()};
  depth_type probcut_depth_{fixed().probcut_depth()};
  depth_type iir_depth_{fixed().iir_depth()};

  score_type aspiration_delta_{fixed().aspiration_delta()};

  see_type nmp_see_threshold_{fixed().nmp_see_threshold()};
  depth_type singular_extension_depth_margin_{fixed().singular_extension_depth_margin()};
  score_type singular_double_extension_margin_{fixed().singular_double_extension_margin()};

  score_type nmp_reduction_depth_b_{3};
  score_type nmp_reduction_depth_div_{4};
  score_type nmp_reduction_eval_delta_div_{242};
  score_type nmp_reduction_eval_delta_based_depth_limit_{4};

  score_type futility_margin_m_{1544};

  score_type snmp_margin_depth_m_{297};
  score_type snmp_margin_not_threats_and_improving_m_{-297};
  score_type snmp_margin_improving_m_{0};
  score_type snmp_margin_not_threats_m_{-112};
  score_type snmp_margin_b_{112};

  see_type quiet_see_prune_threshold_m_{-54};
  see_type noisy_see_prune_threshold_m_{-111};

  counter_type history_prune_threshold_m_{-1397};
  counter_type history_reduction_div_{5872};

  depth_type base_reduction_offset_{0};
  depth_type improving_reduction_offset_{reduction_offset_scale};
  depth_type is_check_reduction_offset_{reduction_offset_scale};
  depth_type creates_threat_reduction_offset_{reduction_offset_scale};
  depth_type is_killer_reduction_offset_{reduction_offset_scale};
  depth_type not_tt_pv_reduction_offset_{reduction_offset_scale};
  depth_type opponent_reducer_reduction_offset_{reduction_offset_scale};

  score_type delta_margin_{fixed().delta_margin()};

  see_type good_capture_prune_see_margin_{fixed().good_capture_prune_see_margin()};
  score_type good_capture_prune_score_margin_{fixed().good_capture_prune_score_margin()};

  score_type probcut_search_depth_offset_{3};
  score_type probcut_beta_b_{315};

  depth_type razor_depth_{fixed().razor_depth()};
  depth_type razor_margin_m_{896};

  double lmr_b_{0.4711025619218822};
  double lmr_div_{2.5650906222921046};

  [[nodiscard]] constexpr depth_type reduce_depth() const noexcept { return reduce_depth_; }
  [[nodiscard]] constexpr depth_type aspiration_depth() const noexcept { return aspiration_depth_; }
  [[nodiscard]] constexpr depth_type nmp_depth() const noexcept { return nmp_depth_; }

  [[nodiscard]] constexpr depth_type snmp_depth() const noexcept { return snmp_depth_; }
  [[nodiscard]] constexpr depth_type futility_prune_depth() const noexcept { return futility_prune_depth_; }

  [[nodiscard]] constexpr depth_type quiet_see_prune_depth() const noexcept { return quiet_see_prune_depth_; }
  [[nodiscard]] constexpr depth_type noisy_see_prune_depth() const noexcept { return noisy_see_prune_depth_; }

  [[nodiscard]] constexpr depth_type singular_extension_depth() const noexcept { return singular_extension_depth_; }
  [[nodiscard]] constexpr depth_type probcut_depth() const noexcept { return probcut_depth_; }
  [[nodiscard]] constexpr depth_type iir_depth() const noexcept { return iir_depth_; }

  [[nodiscard]] constexpr score_type aspiration_delta() const noexcept { return aspiration_delta_; }

  [[nodiscard]] constexpr depth_type nmp_reduction(const depth_type& depth, const score_type& beta, const score_type& value) const noexcept {
    return nmp_reduction_depth_b_ + depth / nmp_reduction_depth_div_ +
           std::min(nmp_reduction_eval_delta_based_depth_limit_, (value - beta) / nmp_reduction_eval_delta_div_);
  }

  [[nodiscard]] constexpr see_type nmp_see_threshold() const noexcept { return nmp_see_threshold_; }

  [[nodiscard]] constexpr depth_type singular_extension_depth_margin() const noexcept { return singular_extension_depth_margin_; }
  [[nodiscard]] constexpr score_type singular_double_extension_margin() const noexcept { return singular_double_extension_margin_; }

  [[nodiscard]] constexpr score_type futility_margin(const depth_type& depth) const noexcept {
    return futility_margin_m_ * static_cast<score_type>(depth);
  }

  [[nodiscard]] constexpr score_type snmp_margin(const bool& improving, const bool& threats, const depth_type& depth) const noexcept {
    return snmp_margin_depth_m_ * depth + snmp_margin_not_threats_and_improving_m_ * (improving && !threats) + snmp_margin_improving_m_ * improving +
           snmp_margin_not_threats_m_ * !threats + snmp_margin_b_;
  }

  [[nodiscard]] constexpr see_type quiet_see_prune_threshold(const depth_type& depth) const noexcept {
    return quiet_see_prune_threshold_m_ * static_cast<see_type>(depth);
  }
  [[nodiscard]] constexpr see_type noisy_see_prune_threshold(const depth_type& depth) const noexcept {
    return noisy_see_prune_threshold_m_ * static_cast<see_type>(depth);
  }

  [[nodiscard]] constexpr counter_type history_prune_threshold(const depth_type& depth) const noexcept {
    return history_prune_threshold_m_ * static_cast<counter_type>(depth * depth);
  }

  [[nodiscard]] constexpr int lmp_count(const bool& improving, const depth_type& depth) const noexcept {
    const std::array<int, 8> improving_counts = {
        0, lmp_improving_1_, lmp_improving_2_, lmp_improving_3_, lmp_improving_4_, lmp_improving_5_, lmp_improving_6_, lmp_improving_7_};
    const std::array<int, 8> worsening_counts = {
        0, lmp_worsening_1_, lmp_worsening_2_, lmp_worsening_3_, lmp_worsening_4_, lmp_worsening_5_, lmp_worsening_6_, lmp_worsening_7_};
    return improving ? improving_counts[depth] : worsening_counts[depth];
  }

  [[nodiscard]] constexpr depth_type history_reduction(const counter_type& history_value) const noexcept {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / history_reduction_div_);
    return std::clamp(raw, -limit, limit);
  }

  [[nodiscard]] constexpr depth_type base_reduction_offset() const noexcept { return base_reduction_offset_; }
  [[nodiscard]] constexpr depth_type improving_reduction_offset() const noexcept { return improving_reduction_offset_; }
  [[nodiscard]] constexpr depth_type is_check_reduction_offset() const noexcept { return is_check_reduction_offset_; }
  [[nodiscard]] constexpr depth_type creates_threat_reduction_offset() const noexcept { return creates_threat_reduction_offset_; }
  [[nodiscard]] constexpr depth_type is_killer_reduction_offset() const noexcept { return is_killer_reduction_offset_; }
  [[nodiscard]] constexpr depth_type not_tt_pv_reduction_offset() const noexcept { return not_tt_pv_reduction_offset_; }
  [[nodiscard]] constexpr depth_type opponent_reducer_reduction_offset() const noexcept { return opponent_reducer_reduction_offset_; }

  [[nodiscard]] constexpr score_type delta_margin() const noexcept { return delta_margin_; }

  [[nodiscard]] constexpr see_type good_capture_prune_see_margin() const noexcept { return good_capture_prune_see_margin_; }
  [[nodiscard]] constexpr score_type good_capture_prune_score_margin() const noexcept { return good_capture_prune_score_margin_; }

  [[nodiscard]] constexpr depth_type probcut_search_depth(const depth_type& depth) const noexcept {
    return std::max(0, depth - probcut_search_depth_offset_);
  }

  [[nodiscard]] constexpr score_type probcut_beta(const score_type& beta) const noexcept { return beta + probcut_beta_b_; }

  [[nodiscard]] constexpr depth_type razor_depth() const noexcept { return razor_depth_; }
  [[nodiscard]] constexpr depth_type razor_margin(const depth_type& depth) const noexcept { return razor_margin_m_ * depth; }

  [[nodiscard]] constexpr depth_type reduction(const depth_type& depth, const int& move_idx) const noexcept {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }

  [[maybe_unused]] tuning_search_constants& update_(const std::size_t& thread_count) noexcept {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(lmr_b_ + std::log(depth) * std::log(played) / lmr_div_);
      }
    }
    return *this;
  }

  auto options() noexcept {
    // clang-format off
    return engine::uci_options(
      INTEGRAL_OPTION(reduce_depth_, 1, 5, 1, 0.002),
      INTEGRAL_OPTION(aspiration_depth_, 1, 7, 1, 0.002),
      INTEGRAL_OPTION(nmp_depth_, 1, 5, 1, 0.002),
      INTEGRAL_OPTION(snmp_depth_, 3, 11, 1, 0.002),
      INTEGRAL_OPTION(futility_prune_depth_, 5, 11, 1, 0.002),
      INTEGRAL_OPTION(quiet_see_prune_depth_, 4, 13, 1, 0.002),
      INTEGRAL_OPTION(noisy_see_prune_depth_, 4, 13, 1, 0.002),
      INTEGRAL_OPTION(singular_extension_depth_, 4, 10, 1, 0.002),
      INTEGRAL_OPTION(probcut_depth_, 4, 8, 1, 0.002),
      INTEGRAL_OPTION(iir_depth_, 2, 5, 1, 0.002),

      INTEGRAL_OPTION(aspiration_delta_, 5, 35, 4, 0.002),
    
      INTEGRAL_OPTION(nmp_see_threshold_, 150, 1000, 25, 0.002),
      INTEGRAL_OPTION(nmp_reduction_depth_b_, 2, 11, 1, 0.002),
      INTEGRAL_OPTION(nmp_reduction_depth_div_, 3, 9, 1, 0.002),
      INTEGRAL_OPTION(nmp_reduction_eval_delta_div_, 100, 400, 10, 0.002),
      INTEGRAL_OPTION(nmp_reduction_eval_delta_based_depth_limit_, 2, 5, 1, 0.002),

      INTEGRAL_OPTION(singular_extension_depth_margin_, 1, 5, 1, 0.002),
      INTEGRAL_OPTION(singular_double_extension_margin_, 70, 500, 10, 0.002),

      INTEGRAL_OPTION(futility_margin_m_, 500, 2500, 10, 0.002),
      INTEGRAL_OPTION(snmp_margin_depth_m_, 100, 400, 10, 0.002),
      INTEGRAL_OPTION(snmp_margin_not_threats_and_improving_m_, -512, 512, 10, 0.002),
      INTEGRAL_OPTION(snmp_margin_not_threats_m_, -512, 512, 10, 0.002),
      INTEGRAL_OPTION(snmp_margin_improving_m_, -512, 512, 10, 0.002),
      INTEGRAL_OPTION(snmp_margin_b_, 25, 250, 10, 0.002),
      
      INTEGRAL_OPTION(quiet_see_prune_threshold_m_, -200, -25, 25, 0.002),
      INTEGRAL_OPTION(noisy_see_prune_threshold_m_, -400, -100, 25, 0.002),
      INTEGRAL_OPTION(history_prune_threshold_m_, -2048, -512, 250, 0.002),
      
      INTEGRAL_OPTION(lmp_improving_1_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_improving_2_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_improving_3_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_improving_4_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_improving_5_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_improving_6_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_improving_7_, 2, 75, 1, 0.002),

      INTEGRAL_OPTION(lmp_worsening_1_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_worsening_2_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_worsening_3_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_worsening_4_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_worsening_5_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_worsening_6_, 2, 75, 1, 0.002),
      INTEGRAL_OPTION(lmp_worsening_7_, 2, 75, 1, 0.002),

      INTEGRAL_OPTION(history_reduction_div_, 4096, 8192, 450, 0.002),
      INTEGRAL_OPTION(base_reduction_offset_, -1024, 1024, 384, 0.002),
      INTEGRAL_OPTION(improving_reduction_offset_, 0, 2048, 384, 0.002),
      INTEGRAL_OPTION(is_check_reduction_offset_, 0, 2048, 384, 0.002),
      INTEGRAL_OPTION(creates_threat_reduction_offset_, 0, 2048, 384, 0.002),
      INTEGRAL_OPTION(is_killer_reduction_offset_, 0, 2048, 384, 0.002),
      INTEGRAL_OPTION(not_tt_pv_reduction_offset_, 0, 2048, 384, 0.002),
      INTEGRAL_OPTION(opponent_reducer_reduction_offset_, 0, 2048, 384, 0.002),

      INTEGRAL_OPTION(delta_margin_, 256, 1024, 10, 0.002),

      INTEGRAL_OPTION(good_capture_prune_see_margin_, 150, 1000, 50, 0.002),
      INTEGRAL_OPTION(good_capture_prune_score_margin_, 128, 1024, 10, 0.002),
      
      INTEGRAL_OPTION(probcut_search_depth_offset_, 1, 7, 1, 0.002),
      INTEGRAL_OPTION(probcut_beta_b_, 100, 1000, 10, 0.002),

      INTEGRAL_OPTION(razor_depth_, 1, 5, 1, 0.002),
      INTEGRAL_OPTION(razor_margin_m_, 512, 1024, 10, 0.002),

      FLOATING_OPTION(lmr_b_, 0.0, 2.5, 0.1, 0.002),
      FLOATING_OPTION(lmr_div_, 1.0, 3.0, 0.1, 0.002)
    );
    // clang-format on
  }

  explicit tuning_search_constants(const std::size_t& thread_count = 1) noexcept { update_(thread_count); }
};

using search_constants = tuning_search_constants;

}  // namespace search
