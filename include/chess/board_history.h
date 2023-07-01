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

#include <chess/move.h>
#include <zobrist/util.h>

#include <algorithm>
#include <vector>

namespace chess {

template <typename T, typename U>
struct base_history {
  using value_type = U;
  std::vector<value_type> history_;

  [[nodiscard]] constexpr T& cast() noexcept { return static_cast<T&>(*this); }
  [[nodiscard]] const T& cast() const noexcept { return static_cast<const T&>(*this); }

  [[maybe_unused]] T& clear() noexcept {
    history_.clear();
    return cast();
  }

  [[maybe_unused]] T& push(const value_type& elem) noexcept {
    history_.push_back(elem);
    return cast();
  }

  base_history() noexcept : history_{} {}
  base_history(const std::vector<value_type>& history) noexcept : history_{history} {}
};

struct board_history : base_history<board_history, zobrist::hash_type> {
  [[nodiscard]] std::size_t count(const zobrist::hash_type& hash) const noexcept { return std::count(history_.crbegin(), history_.crend(), hash); }
};

}  // namespace chess
