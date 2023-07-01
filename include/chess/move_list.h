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

#include <chess/castle_info.h>
#include <chess/move.h>
#include <chess/pawn_info.h>
#include <chess/square.h>
#include <chess/types.h>
#include <util/bit_range.h>

#include <algorithm>
#include <array>
#include <iostream>
#include <utility>

namespace chess {

struct move_list {
  static constexpr std::size_t max_branching_factor = 192;
  using iterator = std::array<move, max_branching_factor>::iterator;
  using const_iterator = std::array<move, max_branching_factor>::const_iterator;

  std::size_t size_{0};
  std::array<move, max_branching_factor> data{};

  [[nodiscard]] constexpr iterator begin() noexcept { return data.begin(); }
  [[nodiscard]] constexpr iterator end() noexcept { return data.begin() + size_; }
  [[nodiscard]] constexpr const_iterator begin() const noexcept { return data.cbegin(); }
  [[nodiscard]] constexpr const_iterator end() const noexcept { return data.cbegin() + size_; }

  [[nodiscard]] inline bool has(const move& mv) const noexcept { return end() != std::find(begin(), end(), mv); }

  [[nodiscard]] constexpr std::size_t size() const noexcept { return size_; }
  [[nodiscard]] constexpr bool empty() const noexcept { return size_ == 0; }

  [[nodiscard]] constexpr move& operator[](const std::size_t& idx) noexcept { return data[idx]; }
  [[nodiscard]] constexpr const move& operator[](const std::size_t& idx) const noexcept { return data[idx]; }

  [[maybe_unused]] constexpr move_list& push(const move& mv) noexcept {
    constexpr std::size_t last_idx = max_branching_factor - 1;
    data[size_] = mv;
    ++size_;
    if (size_ > last_idx) { size_ = last_idx; }
    return *this;
  }

  template <typename... Ts>
  [[maybe_unused]] constexpr move_list& push(const Ts&... ts) noexcept {
    return push(move(ts...));
  }

  template <typename... Ts>
  [[maybe_unused]] constexpr move_list& push_queen_promotion(const Ts&... ts) noexcept {
    push(move(ts...).set_field_<move::promotion_>(piece_type::queen));
    return *this;
  }

  template <typename... Ts>
  [[maybe_unused]] constexpr move_list& push_under_promotions(const Ts&... ts) noexcept {
    for (const auto& pt : under_promotion_types) { push(move(ts...).set_field_<move::promotion_>(pt)); }
    return *this;
  }
};

std::ostream& operator<<(std::ostream& ostr, const move_list& mv_ls) noexcept;

}  // namespace chess