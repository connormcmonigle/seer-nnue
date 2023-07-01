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

#include <chess/move.h>

namespace chess {

template <color c>
std::string move::name() const noexcept {
  if (is_castle_oo<c>()) {
    return castle_info<c>.start_king.name() + castle_info<c>.after_oo_king.name();
  } else if (is_castle_ooo<c>()) {
    return castle_info<c>.start_king.name() + castle_info<c>.after_ooo_king.name();
  }

  std::string base = from().name() + to().name();
  if (is_promotion<c>()) {
    return base + piece_letter(promotion());
  } else {
    return base;
  }
}

std::string move::name(const bool pov) const noexcept { return pov ? name<color::white>() : name<color::black>(); }

std::ostream& operator<<(std::ostream& ostr, const move& mv) noexcept {
  ostr << "move(from=" << mv.from().name() << ", to=" << mv.to().name() << ", piece=" << piece_name(mv.piece()) << ", is_capture=" << mv.is_capture()
       << ", capture=" << piece_name(mv.captured()) << ", is_enpassant=" << mv.is_enpassant() << ", enpassant_sq=" << mv.enpassant_sq().name()
       << ", promotion=" << piece_name(mv.promotion()) << ')';
  return ostr;
}

}  // namespace chess