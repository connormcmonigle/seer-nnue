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

#include <cctype>
#include <string_view>
#include <type_traits>

namespace chess {

enum class color : std::uint8_t { white, black };

template <color>
struct them_ {};

template <>
struct them_<color::white> {
  static constexpr color value = color::black;
};

template <>
struct them_<color::black> {
  static constexpr color value = color::white;
};

template <typename T, typename U>
struct sided {
  using return_type = U;

  T& cast() { return static_cast<T&>(*this); }
  const T& cast() const { return static_cast<const T&>(*this); }

  template <color c>
  return_type& us() {
    if constexpr (c == color::white) {
      return cast().white;
    } else {
      return cast().black;
    }
  }

  template <color c>
  const return_type& us() const {
    if constexpr (c == color::white) {
      return cast().white;
    } else {
      return cast().black;
    }
  }

  template <color c>
  return_type& them() {
    return us<them_<c>::value>();
  }

  template <color c>
  const return_type& them() const {
    return us<them_<c>::value>();
  }

  return_type& us(bool side) { return side ? us<color::white>() : us<color::black>(); }

  const return_type& us(bool side) const { return side ? us<color::white>() : us<color::black>(); }

  return_type& them(bool side) { return us(!side); }

  const return_type& them(bool side) const { return us(!side); }

  return_type& us(color side) { return us(side == color::white); }

  const return_type& us(color side) const { return us(side == color::white); }

  return_type& them(color side) { return us(side != color::white); }

  const return_type& them(color side) const { return us(side != color::white); }

 private:
  sided(){};
  friend T;
};

color color_from(char ch) { return std::isupper(ch) ? color::white : color::black; }

enum class piece_type : std::uint8_t { pawn, knight, bishop, rook, queen, king };

piece_type type_from(const char& ch) {
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

constexpr char piece_letter(const piece_type& p) {
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

constexpr char piece_letter(const color& c, const piece_type& p) {
  const char p_letter = piece_letter(p);
  switch (c) {
    case color::white: return std::toupper(p_letter);
    case color::black: return std::tolower(p_letter);
    default: return p_letter;
  }
}

constexpr std::string_view piece_name(const piece_type& p) {
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
constexpr void over_types(F&& f) {
  f(piece_type::king);
  f(piece_type::queen);
  f(piece_type::rook);
  f(piece_type::bishop);
  f(piece_type::knight);
  f(piece_type::pawn);
}

template <typename T>
auto get_member(const piece_type& idx, T& set) -> decltype(set.pawn_)& {
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
auto get_member(const piece_type& idx, const T& set) -> const decltype(set.pawn_)& {
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
