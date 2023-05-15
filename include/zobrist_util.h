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

#include <array>
#include <cstdint>
#include <limits>
#include <random>

namespace zobrist {

using hash_type = std::uint64_t;
using half_hash_type = std::uint32_t;

constexpr std::mt19937::result_type seed = 0x019ec6dc;

constexpr half_hash_type lower_half(const hash_type& hash) { return hash & std::numeric_limits<half_hash_type>::max(); }
constexpr half_hash_type upper_half(const hash_type& hash) { return (hash >> 32) & std::numeric_limits<half_hash_type>::max(); }

inline hash_type random_bit_string() {
  static std::mt19937 gen(seed);
  constexpr hash_type a = std::numeric_limits<hash_type>::min();
  constexpr hash_type b = std::numeric_limits<hash_type>::max();
  std::uniform_int_distribution<hash_type> dist(a, b);
  return dist(gen);
}

struct latent_src {
  static constexpr size_t num_squares = 64;
  zobrist::hash_type oo_;
  zobrist::hash_type ooo_;
  std::array<zobrist::hash_type, num_squares> ep_mask_;

  zobrist::hash_type get_oo() const { return oo_; }
  zobrist::hash_type get_ooo() const { return ooo_; }

  template <typename S>
  zobrist::hash_type get_ep_mask(const S& at) const {
    static_assert(chess::is_square_v<S>, "at must be of square type");
    return ep_mask_[at.index()];
  }

  latent_src() {
    oo_ = zobrist::random_bit_string();
    ooo_ = zobrist::random_bit_string();
    std::transform(ep_mask_.begin(), ep_mask_.end(), ep_mask_.begin(), [](auto...) { return zobrist::random_bit_string(); });
  }
};

struct manifest_src {
  static constexpr size_t num_squares = 64;
  using plane_t = std::array<zobrist::hash_type, num_squares>;
  plane_t pawn_{};
  plane_t knight_{};
  plane_t bishop_{};
  plane_t rook_{};
  plane_t queen_{};
  plane_t king_{};

  plane_t& get_plane(const chess::piece_type& pt) { return chess::get_member(pt, *this); }
  const plane_t& get_plane(const chess::piece_type& pt) const { return chess::get_member(pt, *this); }

  template <typename S>
  zobrist::hash_type get(const chess::piece_type& pt, const S& at) const {
    static_assert(chess::is_square_v<S>, "at must be of square type");
    return get_plane(pt)[at.index()];
  }

  manifest_src() {
    chess::over_types([this](const chess::piece_type pt) {
      plane_t& pt_plane = get_plane(pt);
      std::transform(pt_plane.begin(), pt_plane.end(), pt_plane.begin(), [](auto...) { return zobrist::random_bit_string(); });
    });
  }
};

struct sided_manifest_src : chess::sided<sided_manifest_src, manifest_src> {
  manifest_src white;
  manifest_src black;

  sided_manifest_src() : white{}, black{} {}
};

struct sided_latent_src : chess::sided<sided_latent_src, latent_src> {
  latent_src white;
  latent_src black;

  sided_latent_src() : white{}, black{} {}
};

struct sided_turn_src : chess::sided<sided_turn_src, hash_type> {
  hash_type white;
  hash_type black;

  sided_turn_src() : white{zobrist::random_bit_string()}, black{zobrist::random_bit_string()} {}
};

struct sources {
  static inline const sided_manifest_src manifest{};
  static inline const sided_latent_src latent{};
  static inline const sided_turn_src turn{};
};

}  // namespace zobrist
