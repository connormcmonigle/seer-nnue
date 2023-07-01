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

#include <chess/types.h>
#include <chess/square.h>

namespace chess {

struct piece_configuration {
  square_set pawn_{};
  square_set knight_{};
  square_set bishop_{};
  square_set rook_{};
  square_set queen_{};
  square_set king_{};

  [[nodiscard]] constexpr const square_set& get_plane(const piece_type& pt) const noexcept { return get_member(pt, *this); }
  constexpr void set_plane(const piece_type& pt, const square_set& plane) noexcept { get_member(pt, *this) = plane; }
};

struct sided_piece_configuration : sided<sided_piece_configuration, piece_configuration> {
  piece_configuration white;
  piece_configuration black;

  constexpr sided_piece_configuration() noexcept : white{}, black{} {}
};

}  // namespace chess
