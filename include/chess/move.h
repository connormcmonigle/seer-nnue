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
#include <chess/pawn_info.h>
#include <chess/square.h>
#include <chess/types.h>
#include <util/bit_range.h>

#include <array>
#include <iostream>

namespace chess {

inline constexpr std::array<piece_type, 3> under_promotion_types = {piece_type::knight, piece_type::bishop, piece_type::rook};

struct move {
  using from_ = util::bit_range<std::uint8_t, 0, 6>;
  using to_ = util::next_bit_range<from_, std::uint8_t, 6>;
  using piece_ = util::next_bit_range<to_, piece_type, 3>;
  using is_capture_ = util::next_bit_flag<piece_>;
  using is_enpassant_ = util::next_bit_flag<is_capture_>;
  using captured_ = util::next_bit_range<is_enpassant_, piece_type, 3>;
  using enpassant_sq_ = util::next_bit_range<captured_, std::uint8_t, 6>;
  using promotion_ = util::next_bit_range<enpassant_sq_, piece_type, 3>;

  static constexpr std::size_t width = promotion_::last;
  using data_type = std::uint32_t;

  data_type data;

  template <typename B>
  [[nodiscard]] constexpr typename B::type get_field_() const noexcept {
    return B::get(data);
  }

  template <typename B>
  [[maybe_unused]] constexpr move& set_field_(const typename B::type info) noexcept {
    B::set(data, info);
    return *this;
  }

  [[nodiscard]] constexpr square from() const noexcept { return square::from_index(get_field_<from_>()); }
  [[nodiscard]] constexpr square to() const noexcept { return square::from_index(get_field_<to_>()); }
  [[nodiscard]] constexpr piece_type piece() const noexcept { return get_field_<piece_>(); }
  [[nodiscard]] constexpr bool is_capture() const noexcept { return get_field_<is_capture_>(); }
  [[nodiscard]] constexpr bool is_enpassant() const noexcept { return get_field_<is_enpassant_>(); }
  [[nodiscard]] constexpr piece_type captured() const noexcept { return get_field_<captured_>(); }

  template <typename T>
  [[nodiscard]] constexpr T mvv_lva_key() const noexcept {
    constexpr T num_pieces = static_cast<T>(6);
    return num_pieces * static_cast<T>(get_field_<captured_>()) + num_pieces - static_cast<T>(get_field_<piece_>());
  }

  [[nodiscard]] constexpr square enpassant_sq() const noexcept { return square::from_index(get_field_<enpassant_sq_>()); }
  [[nodiscard]] constexpr piece_type promotion() const noexcept { return get_field_<promotion_>(); }

  [[nodiscard]] constexpr bool is_null() const noexcept { return data == 0; }
  [[nodiscard]] constexpr bool is_king_move() const noexcept { return piece() == piece_type::king; }

  template <color c>
  [[nodiscard]] constexpr bool is_castle_oo() const noexcept {
    return piece() == piece_type::king && from() == castle_info<c>.start_king && to() == castle_info<c>.oo_rook;
  }

  template <color c>
  [[nodiscard]] constexpr bool is_castle_ooo() const noexcept {
    return piece() == piece_type::king && from() == castle_info<c>.start_king && to() == castle_info<c>.ooo_rook;
  }

  template <color c>
  [[nodiscard]] constexpr bool is_promotion() const noexcept {
    return piece() == piece_type::pawn && pawn_info<c>::last_rank.is_member(to());
  }

  [[nodiscard]] constexpr bool is_promotion() const noexcept { return is_promotion<color::white>() || is_promotion<color::black>(); }

  template <color c>
  [[nodiscard]] constexpr bool is_pawn_double() const noexcept {
    return piece() == piece_type::pawn && pawn_info<c>::start_rank.is_member(from()) && pawn_info<c>::double_rank.is_member(to());
  }

  [[nodiscard]] constexpr bool is_quiet() const noexcept { return !is_capture() && !(is_promotion() && piece_type::queen == promotion()); }
  [[nodiscard]] constexpr bool is_noisy() const noexcept { return !is_quiet(); }

  template <color c>
  [[nodiscard]] std::string name() const noexcept;
  [[nodiscard]] std::string name(bool pov) const noexcept;

  move() = default;

  constexpr explicit move(const data_type& data) noexcept : data{data} {}

  constexpr move(
      square from,
      square to,
      piece_type piece,
      bool is_capture = false,
      piece_type captured = piece_type::pawn,
      bool is_enpassant = false,
      square enpassant_sq = square::from_index(0),
      piece_type promotion = piece_type::pawn) noexcept
      : data{0} {
    const auto from_idx = static_cast<std::uint8_t>(from.index());
    const auto to_idx = static_cast<std::uint8_t>(to.index());
    const auto ep_sq_idx = static_cast<std::uint8_t>(enpassant_sq.index());
    set_field_<from_>(from_idx)
        .set_field_<to_>(to_idx)
        .set_field_<piece_>(piece)
        .set_field_<is_capture_>(is_capture)
        .set_field_<is_enpassant_>(is_enpassant)
        .set_field_<captured_>(captured)
        .set_field_<enpassant_sq_>(ep_sq_idx)
        .set_field_<promotion_>(promotion);
  }

  constexpr static move null() noexcept { return move{0}; }
};

constexpr bool operator==(const move& a, const move& b) { return a.data == b.data; }
constexpr bool operator!=(const move& a, const move& b) { return !(a == b); }
std::ostream& operator<<(std::ostream& ostr, const move& mv) noexcept;

struct move_hash {
  [[nodiscard]] std::size_t operator()(const move& mv) const noexcept { return std::hash<std::uint32_t>{}(mv.data); }
};

}  // namespace chess
