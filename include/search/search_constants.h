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

inline constexpr depth_type safe_depth = max_depth + max_depth_margin;

inline constexpr depth_type reduction_offset_scale = 1024;

using score_type = std::int32_t;

using wdl_type = std::tuple<score_type, score_type, score_type>;

inline constexpr score_type big_number = 8 * logit_scale<score_type>;

inline constexpr score_type max_mate_score = -2 * big_number;

inline constexpr score_type mate_score = max_mate_score - (max_depth + max_depth_margin);

inline constexpr score_type tb_win_score = big_number + 1;

inline constexpr score_type tb_loss_score = -tb_win_score;

inline constexpr score_type draw_score = 0;

inline constexpr score_type aspiration_delta = 23;

using counter_type = std::int32_t;

using see_type = std::int32_t;

inline constexpr std::size_t nodes_per_update = 512;

struct fixed_search_constants final {
  static constexpr bool tuning = false;
  static constexpr depth_type lmr_tbl_dim = 64;
  std::size_t thread_count_;
  std::array<depth_type, lmr_tbl_dim * lmr_tbl_dim> lmr_tbl{};

  [[nodiscard]] const std::size_t& thread_count() const noexcept { return thread_count_; }

  [[nodiscard]] constexpr depth_type reduce_depth() const noexcept { return 2; }
  [[nodiscard]] constexpr depth_type aspiration_depth() const noexcept { return 3; }
  [[nodiscard]] constexpr depth_type nmp_depth() const noexcept { return 4; }
  [[nodiscard]] constexpr depth_type lmp_depth() const noexcept { return 7; }
  [[nodiscard]] constexpr depth_type snmp_depth() const noexcept { return 6; }
  [[nodiscard]] constexpr depth_type futility_prune_depth() const noexcept { return 5; }
  [[nodiscard]] constexpr depth_type quiet_see_prune_depth() const noexcept { return 9; }
  [[nodiscard]] constexpr depth_type noisy_see_prune_depth() const noexcept { return 7; }
  [[nodiscard]] constexpr depth_type singular_extension_depth() const noexcept { return 6; }
  [[nodiscard]] constexpr depth_type probcut_depth() const noexcept { return 6; }
  [[nodiscard]] constexpr depth_type iir_depth() const noexcept { return 2; }

  [[nodiscard]] constexpr depth_type reduction(const depth_type& depth, const int& move_idx) const noexcept {
    constexpr depth_type last_idx = lmr_tbl_dim - 1;
    return lmr_tbl[std::min(last_idx, depth) * lmr_tbl_dim + std::min(last_idx, move_idx)];
  }

  [[nodiscard]] constexpr depth_type nmp_reduction(const depth_type& depth, const score_type& beta, const score_type& value) const noexcept {
    return 4 + depth / 3 + std::min(3, (value - beta) / 238);
  }

  [[nodiscard]] constexpr see_type nmp_see_threshold() const noexcept { return 229; }

  [[nodiscard]] constexpr depth_type singular_extension_depth_margin() const noexcept { return 4; }

  [[nodiscard]] constexpr depth_type singular_search_depth(const depth_type& depth) const noexcept { return depth / 2 - 1; }

  [[nodiscard]] constexpr score_type singular_beta(const score_type& tt_score, const depth_type& depth) const noexcept {
    return tt_score - 2 * static_cast<score_type>(depth);
  }

  [[nodiscard]] constexpr score_type singular_double_extension_margin() const noexcept { return 161; }

  [[nodiscard]] constexpr score_type futility_margin(const depth_type& depth) const noexcept {
    constexpr score_type m = 1551;
    return m * static_cast<score_type>(depth);
  }

  [[nodiscard]] constexpr score_type snmp_margin(const bool& improving, const bool& threats, const depth_type& depth) const noexcept {
    constexpr score_type depth_m = 298;
    constexpr score_type not_threats_and_improving_m = -296;
    constexpr score_type not_threats_m = -109;
    constexpr score_type improving_m = 7;
    constexpr score_type b = 118;

    return depth_m * depth + not_threats_and_improving_m * (improving && !threats) + improving_m * improving + not_threats_m * !threats + b;
  }

  [[nodiscard]] constexpr int lmp_count(const bool& improving, const depth_type& depth) const noexcept {
    constexpr std::array<int, 8> improving_counts = {0, 6, 9, 10, 19, 30, 42, 65};
    constexpr std::array<int, 8> worsening_counts = {0, 2, 4, 7, 9, 14, 22, 31};
    return improving ? improving_counts[depth] : worsening_counts[depth];
  }

  [[nodiscard]] constexpr see_type quiet_see_prune_threshold(const depth_type& depth) const noexcept { return -39 * static_cast<see_type>(depth); }
  [[nodiscard]] constexpr see_type noisy_see_prune_threshold(const depth_type& depth) const noexcept { return -137 * static_cast<see_type>(depth); }

  [[nodiscard]] constexpr counter_type history_prune_threshold(const depth_type& depth) const noexcept {
    return -1257 * static_cast<counter_type>(depth * depth);
  }

  [[nodiscard]] constexpr depth_type history_reduction(const counter_type& history_value) const noexcept {
    constexpr depth_type limit = 2;
    const depth_type raw = -static_cast<depth_type>(history_value / 6234);
    return std::clamp(raw, -limit, limit);
  }

  [[nodiscard]] constexpr depth_type base_reduction_offset() const noexcept { return 226; }
  [[nodiscard]] constexpr depth_type improving_reduction_offset() const noexcept { return 1130; }
  [[nodiscard]] constexpr depth_type is_check_reduction_offset() const noexcept { return 1232; }
  [[nodiscard]] constexpr depth_type creates_threat_reduction_offset() const noexcept { return 1084; }
  [[nodiscard]] constexpr depth_type is_killer_reduction_offset() const noexcept { return 832; }
  [[nodiscard]] constexpr depth_type not_tt_pv_reduction_offset() const noexcept { return 1148; }
  [[nodiscard]] constexpr depth_type opponent_reducer_reduction_offset() const noexcept { return 1252; }

  [[nodiscard]] constexpr score_type delta_margin() const noexcept {
    constexpr score_type margin = 496;
    return margin;
  }

  [[nodiscard]] constexpr see_type good_capture_prune_see_margin() const noexcept { return 280; }
  [[nodiscard]] constexpr score_type good_capture_prune_score_margin() const noexcept { return 258; }

  [[nodiscard]] constexpr depth_type probcut_search_depth(const depth_type& depth) const noexcept { return depth - 3; }
  [[nodiscard]] constexpr score_type probcut_beta(const score_type& beta) const noexcept { return beta + 307; }

  [[nodiscard]] constexpr depth_type razor_depth() const noexcept { return 3; }
  [[nodiscard]] constexpr score_type razor_margin(const depth_type& depth) const noexcept { return 895 * depth; }

  [[maybe_unused]] fixed_search_constants& update_(const std::size_t& thread_count) noexcept {
    thread_count_ = thread_count;
    for (depth_type depth{1}; depth < lmr_tbl_dim; ++depth) {
      for (depth_type played{1}; played < lmr_tbl_dim; ++played) {
        lmr_tbl[depth * lmr_tbl_dim + played] = static_cast<depth_type>(0.34095409302904595 + std::log(depth) * std::log(played) / 2.764661846064619);
      }
    }

    return *this;
  }

  explicit fixed_search_constants(const std::size_t& thread_count = 1) noexcept { update_(thread_count); }
};

using search_constants = fixed_search_constants;

}  // namespace search
