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

#include <cstddef>

namespace feature {
namespace half_ka {

constexpr std::size_t numel = 64 * 12 * 64;
constexpr std::size_t max_active_half_features = 32;

constexpr std::size_t major = 64 * 12;
constexpr std::size_t minor = 64;

constexpr std::size_t us_pawn_offset = 0;
constexpr std::size_t us_knight_offset = us_pawn_offset + minor;
constexpr std::size_t us_bishop_offset = us_knight_offset + minor;
constexpr std::size_t us_rook_offset = us_bishop_offset + minor;
constexpr std::size_t us_queen_offset = us_rook_offset + minor;
constexpr std::size_t us_king_offset = us_queen_offset + minor;

constexpr std::size_t them_pawn_offset = us_king_offset + minor;
constexpr std::size_t them_knight_offset = them_pawn_offset + minor;
constexpr std::size_t them_bishop_offset = them_knight_offset + minor;
constexpr std::size_t them_rook_offset = them_bishop_offset + minor;
constexpr std::size_t them_queen_offset = them_rook_offset + minor;
constexpr std::size_t them_king_offset = them_queen_offset + minor;

template <chess::color us>
constexpr int mirror_constant = (chess::color::white == us) ? 0 : 56;

[[nodiscard]] constexpr std::size_t us_offset(const chess::piece_type& pt) noexcept {
  switch (pt) {
    case chess::piece_type::pawn: return us_pawn_offset;
    case chess::piece_type::knight: return us_knight_offset;
    case chess::piece_type::bishop: return us_bishop_offset;
    case chess::piece_type::rook: return us_rook_offset;
    case chess::piece_type::queen: return us_queen_offset;
    case chess::piece_type::king: return us_king_offset;
    default: return us_pawn_offset;
  }
}

[[nodiscard]] constexpr std::size_t them_offset(const chess::piece_type& pt) noexcept {
  switch (pt) {
    case chess::piece_type::pawn: return them_pawn_offset;
    case chess::piece_type::knight: return them_knight_offset;
    case chess::piece_type::bishop: return them_bishop_offset;
    case chess::piece_type::rook: return them_rook_offset;
    case chess::piece_type::queen: return them_queen_offset;
    case chess::piece_type::king: return them_king_offset;
    default: return them_pawn_offset;
  }
}

template <chess::color a, chess::color b>
[[nodiscard]] constexpr std::size_t offset(const chess::piece_type& pt) noexcept {
  return (a == b) ? us_offset(pt) : them_offset(pt);
}

template <chess::color us, chess::color p>
[[nodiscard]] constexpr std::size_t index(const chess::square& ks, const chess::piece_type& pt, const chess::square& sq) noexcept {
  return major * (ks.index() ^ mirror_constant<us>)+offset<us, p>(pt) + (sq.index() ^ mirror_constant<us>);
}

}  // namespace half_ka
}  // namespace feature