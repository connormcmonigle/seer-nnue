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

#include <cctype>
#include <string_view>
#include <type_traits>

namespace chess {

enum class color : std::uint8_t { white, black };

template <color>
struct opponent_impl {};

template <>
struct opponent_impl<color::white> {
  static constexpr color value = color::black;
};

template <>
struct opponent_impl<color::black> {
  static constexpr color value = color::white;
};

template <color c>
constexpr color opponent = opponent_impl<c>::value;

template <typename T, typename U>
struct sided {
  using return_type = U;

  [[nodiscard]] constexpr T& cast() noexcept { return static_cast<T&>(*this); }
  [[nodiscard]] constexpr const T& cast() const noexcept { return static_cast<const T&>(*this); }

  template <color c>
  [[nodiscard]] constexpr return_type& us() noexcept {
    if constexpr (c == color::white) {
      return cast().white;
    } else {
      return cast().black;
    }
  }

  template <color c>
  [[nodiscard]] constexpr const return_type& us() const noexcept {
    if constexpr (c == color::white) {
      return cast().white;
    } else {
      return cast().black;
    }
  }

  template <color c>
  [[nodiscard]] constexpr return_type& them() noexcept {
    return us<opponent<c>>();
  }

  template <color c>
  [[nodiscard]] constexpr const return_type& them() const noexcept {
    return us<opponent<c>>();
  }

  [[nodiscard]] constexpr return_type& us(const bool side) noexcept { return side ? us<color::white>() : us<color::black>(); }

  [[nodiscard]] constexpr const return_type& us(const bool side) const noexcept { return side ? us<color::white>() : us<color::black>(); }

  [[nodiscard]] constexpr return_type& them(const bool side) noexcept { return us(!side); }

  [[nodiscard]] constexpr const return_type& them(const bool side) const noexcept { return us(!side); }

  [[nodiscard]] constexpr return_type& us(const color side) noexcept { return us(side == color::white); }

  [[nodiscard]] constexpr const return_type& us(const color side) const noexcept { return us(side == color::white); }

  [[nodiscard]] constexpr return_type& them(const color side) noexcept { return us(side != color::white); }

  [[nodiscard]] constexpr const return_type& them(const color side) const noexcept { return us(side != color::white); }

 private:
  constexpr sided() noexcept {};
  friend T;
};

[[nodiscard]] inline color color_from(char ch) noexcept { return std::isupper(ch) ? color::white : color::black; }

enum class player_type { white, black, none };

[[nodiscard]] constexpr player_type player_from(const bool& turn) noexcept { return turn ? player_type::white : player_type::black; }

[[nodiscard]] constexpr bool is_player(const player_type& player, const bool& turn) noexcept {
  switch (player) {
    case player_type::white: return turn;
    case player_type::black: return !turn;
    default: return false;
  }
}

enum class piece_type : std::uint8_t { pawn, knight, bishop, rook, queen, king };

[[nodiscard]] inline piece_type type_from(const char& ch) noexcept {
  switch (std::tolower(ch)) {
    case 'p': return piece_type::pawn;
    case 'n': return piece_type::knight;
    case 'b': return piece_type::bishop;
    case 'r': return piece_type::rook;
    case 'q': return piece_type::queen;
    case 'k': return piece_type::king;
    default: return piece_type::king;
  }
}

[[nodiscard]] constexpr char piece_letter(const piece_type& p) noexcept {
  switch (p) {
    case piece_type::pawn: return 'p';
    case piece_type::knight: return 'n';
    case piece_type::bishop: return 'b';
    case piece_type::rook: return 'r';
    case piece_type::queen: return 'q';
    case piece_type::king: return 'k';
    default: return '?';
  }
}

[[nodiscard]] constexpr char piece_letter(const color& c, const piece_type& p) noexcept {
  const char p_letter = piece_letter(p);
  switch (c) {
    case color::white: return std::toupper(p_letter);
    case color::black: return std::tolower(p_letter);
    default: return p_letter;
  }
}

[[nodiscard]] constexpr std::string_view piece_name(const piece_type& p) noexcept {
  switch (p) {
    case piece_type::pawn: return "pawn";
    case piece_type::knight: return "knight";
    case piece_type::bishop: return "bishop";
    case piece_type::rook: return "rook";
    case piece_type::queen: return "queen";
    case piece_type::king: return "king";
    default: return "?";
  }
}

template <typename F>
constexpr void over_types(F&& f) noexcept {
  f(piece_type::king);
  f(piece_type::queen);
  f(piece_type::rook);
  f(piece_type::bishop);
  f(piece_type::knight);
  f(piece_type::pawn);
}

template <typename T>
[[nodiscard]] constexpr auto get_member(const piece_type& idx, T& set) noexcept -> decltype(set.pawn_)& {
  switch (idx) {
    case piece_type::pawn: return set.pawn_;
    case piece_type::knight: return set.knight_;
    case piece_type::bishop: return set.bishop_;
    case piece_type::rook: return set.rook_;
    case piece_type::queen: return set.queen_;
    case piece_type::king: return set.king_;
    default: return set.king_;
  }
}

template <typename T>
[[nodiscard]] constexpr auto get_member(const piece_type& idx, const T& set) noexcept -> const decltype(set.pawn_)& {
  switch (idx) {
    case piece_type::pawn: return set.pawn_;
    case piece_type::knight: return set.knight_;
    case piece_type::bishop: return set.bishop_;
    case piece_type::rook: return set.rook_;
    case piece_type::queen: return set.queen_;
    case piece_type::king: return set.king_;
    default: return set.king_;
  }
}

}  // namespace chess
