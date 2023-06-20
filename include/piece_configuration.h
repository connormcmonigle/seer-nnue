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

#include <chess_types.h>
#include <square.h>

namespace chess {

struct piece_configuration {
  square_set pawn_{};
  square_set knight_{};
  square_set bishop_{};
  square_set rook_{};
  square_set queen_{};
  square_set king_{};

  const square_set& get_plane(const piece_type& pt) const { return get_member(pt, *this); }
  void set_plane(const piece_type& pt, const square_set& plane) { get_member(pt, *this) = plane; }
};

struct sided_piece_configuration : sided<sided_piece_configuration, piece_configuration> {
  piece_configuration white;
  piece_configuration black;

  sided_piece_configuration() : white{}, black{} {}
};

}  // namespace chess