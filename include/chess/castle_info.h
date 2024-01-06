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

#include <chess/square.h>
#include <chess/types.h>

namespace chess {

template <color C>
struct castle_info_ {};

template <>
struct castle_info_<color::white> {
  static constexpr tbl_square oo_rook_tbl{0, 0};
  static constexpr tbl_square ooo_rook_tbl{7, 0};
  static constexpr tbl_square start_king_tbl{3, 0};

  static constexpr tbl_square after_oo_rook_tbl{2, 0};
  static constexpr tbl_square after_ooo_rook_tbl{4, 0};
  static constexpr tbl_square after_oo_king_tbl{1, 0};
  static constexpr tbl_square after_ooo_king_tbl{5, 0};

  square oo_rook;
  square ooo_rook;
  square start_king;

  square after_oo_rook;
  square after_ooo_rook;
  square after_oo_king;
  square after_ooo_king;

  square_set oo_mask;

  square_set ooo_danger_mask;
  square_set ooo_occ_mask;

  constexpr castle_info_() noexcept
      : oo_rook{oo_rook_tbl.to_square()},
        ooo_rook{ooo_rook_tbl.to_square()},
        start_king{start_king_tbl.to_square()},
        after_oo_rook{after_oo_rook_tbl.to_square()},
        after_ooo_rook{after_ooo_rook_tbl.to_square()},
        after_oo_king{after_oo_king_tbl.to_square()},
        after_ooo_king{after_ooo_king_tbl.to_square()} {
    constexpr delta ooo_delta{1, 0};
    constexpr delta oo_delta{-1, 0};

    for (auto sq = start_king_tbl.add(oo_delta); true; sq = sq.add(oo_delta)) {
      oo_mask.insert(sq);
      if (sq == after_oo_king_tbl) { break; }
    }

    for (auto sq = start_king_tbl.add(ooo_delta); true; sq = sq.add(ooo_delta)) {
      ooo_danger_mask.insert(sq);
      if (sq == after_ooo_king_tbl) { break; }
    }

    for (auto sq = start_king_tbl.add(ooo_delta); sq != ooo_rook_tbl; sq = sq.add(ooo_delta)) { ooo_occ_mask.insert(sq); }
  }
};

template <>
struct castle_info_<color::black> {
  static constexpr tbl_square oo_rook_tbl{0, 7};
  static constexpr tbl_square ooo_rook_tbl{7, 7};
  static constexpr tbl_square start_king_tbl{3, 7};

  static constexpr tbl_square after_oo_rook_tbl{2, 7};
  static constexpr tbl_square after_ooo_rook_tbl{4, 7};
  static constexpr tbl_square after_oo_king_tbl{1, 7};
  static constexpr tbl_square after_ooo_king_tbl{5, 7};

  square oo_rook;
  square ooo_rook;
  square start_king;

  square after_oo_rook;
  square after_ooo_rook;
  square after_oo_king;
  square after_ooo_king;

  square_set oo_mask;

  square_set ooo_danger_mask;
  square_set ooo_occ_mask;

  constexpr castle_info_() noexcept
      : oo_rook{oo_rook_tbl.to_square()},
        ooo_rook{ooo_rook_tbl.to_square()},
        start_king{start_king_tbl.to_square()},
        after_oo_rook{after_oo_rook_tbl.to_square()},
        after_ooo_rook{after_ooo_rook_tbl.to_square()},
        after_oo_king{after_oo_king_tbl.to_square()},
        after_ooo_king{after_ooo_king_tbl.to_square()} {
    constexpr delta ooo_delta{1, 0};
    constexpr delta oo_delta{-1, 0};

    for (auto sq = start_king_tbl.add(oo_delta); true; sq = sq.add(oo_delta)) {
      oo_mask.insert(sq);
      if (sq == after_oo_king_tbl) { break; }
    }

    for (auto sq = start_king_tbl.add(ooo_delta); true; sq = sq.add(ooo_delta)) {
      ooo_danger_mask.insert(sq);
      if (sq == after_ooo_king_tbl) { break; }
    }

    for (auto sq = start_king_tbl.add(ooo_delta); sq != ooo_rook_tbl; sq = sq.add(ooo_delta)) { ooo_occ_mask.insert(sq); }
  }
};

template <color c>
inline constexpr castle_info_<c> castle_info = castle_info_<c>{};

}  // namespace chess
