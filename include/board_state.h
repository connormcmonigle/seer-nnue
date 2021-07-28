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

#include <enum_util.h>
#include <move.h>
#include <square.h>
#include <table_generation.h>
#include <zobrist_util.h>

#include <array>
#include <limits>
#include <tuple>
#include <vector>

namespace chess {

struct manifest_zobrist_src {
  static constexpr size_t num_squares = 64;
  using plane_t = std::array<zobrist::hash_type, num_squares>;
  plane_t pawn_{};
  plane_t knight_{};
  plane_t bishop_{};
  plane_t rook_{};
  plane_t queen_{};
  plane_t king_{};

  std::array<zobrist::hash_type, num_squares>& get_plane(const piece_type& pt) { return get_member(pt, *this); }

  const std::array<zobrist::hash_type, num_squares>& get_plane(const piece_type& pt) const { return get_member(pt, *this); }

  template <typename S>
  zobrist::hash_type get(const piece_type& pt, const S& at) const {
    static_assert(is_square_v<S>, "at must be of square type");
    return get_plane(pt)[at.index()];
  }

  manifest_zobrist_src() {
    over_types([this](const piece_type pt) {
      plane_t& pt_plane = get_plane(pt);
      std::transform(pt_plane.begin(), pt_plane.end(), pt_plane.begin(), [](auto...) { return zobrist::random_bit_string(); });
    });
  }
};

struct manifest {
  static constexpr size_t num_squares = 64;

  const manifest_zobrist_src* zobrist_src_;
  zobrist::hash_type hash_{0};
  square_set pawn_{};
  square_set knight_{};
  square_set bishop_{};
  square_set rook_{};
  square_set queen_{};
  square_set king_{};
  square_set all_{};

  zobrist::hash_type hash() const { return hash_; }

  square_set& get_plane(const piece_type pt) { return get_member(pt, *this); }

  piece_type occ(const tbl_square& at) const { return occ(at.to_square()); }

  piece_type occ(const square& at) const {
    if (knight_.is_member(at)) { return piece_type::knight; }
    if (bishop_.is_member(at)) { return piece_type::bishop; }
    if (rook_.is_member(at)) { return piece_type::rook; }
    if (queen_.is_member(at)) { return piece_type::queen; }
    if (king_.is_member(at)) { return piece_type::king; }
    return piece_type::pawn;
  }

  const square_set& all() const { return all_; }
  const square_set& pawn() const { return pawn_; }
  const square_set& knight() const { return knight_; }
  const square_set& bishop() const { return bishop_; }
  const square_set& rook() const { return rook_; }
  const square_set& queen() const { return queen_; }
  const square_set& king() const { return king_; }

  const square_set& get_plane(const piece_type pt) const { return get_member(pt, *this); }

  template <typename S>
  manifest& add_piece(const piece_type& pt, const S& at) {
    static_assert(is_square_v<S>, "at must be of square type");
    hash_ ^= zobrist_src_->get(pt, at);
    all_ |= at.bit_board();
    get_plane(pt) |= at.bit_board();
    return *this;
  }

  template <typename S>
  manifest& remove_piece(const piece_type& pt, const S& at) {
    static_assert(is_square_v<S>, "at must be of square type");
    hash_ ^= zobrist_src_->get(pt, at);
    all_ &= ~at.bit_board();
    get_plane(pt) &= ~at.bit_board();
    return *this;
  }

  manifest(const manifest_zobrist_src* src) : zobrist_src_{src} {}
};

struct sided_manifest : sided<sided_manifest, manifest> {
  static inline const manifest_zobrist_src w_manifest_src{};
  static inline const manifest_zobrist_src b_manifest_src{};

  manifest white;
  manifest black;

  zobrist::hash_type hash() const { return white.hash() ^ black.hash(); }

  sided_manifest() : white(&w_manifest_src), black(&b_manifest_src) {}
};

struct latent_zobrist_src {
  static constexpr size_t num_squares = 64;
  zobrist::hash_type oo_;
  zobrist::hash_type ooo_;
  std::array<zobrist::hash_type, num_squares> ep_mask_;

  zobrist::hash_type get_oo() const { return oo_; }
  zobrist::hash_type get_ooo() const { return ooo_; }

  template <typename S>
  zobrist::hash_type get_ep_mask(const S& at) const {
    static_assert(is_square_v<S>, "at must be of square type");
    return ep_mask_[at.index()];
  }

  latent_zobrist_src() {
    oo_ = zobrist::random_bit_string();
    ooo_ = zobrist::random_bit_string();
    std::transform(ep_mask_.begin(), ep_mask_.end(), ep_mask_.begin(), [](auto...) { return zobrist::random_bit_string(); });
  }
};

struct latent {
  const latent_zobrist_src* zobrist_src_;
  zobrist::hash_type hash_{0};
  bool oo_{true};
  bool ooo_{true};
  square_set ep_mask_{};

  zobrist::hash_type hash() const { return hash_; }

  bool oo() const { return oo_; }

  bool ooo() const { return ooo_; }

  const square_set& ep_mask() const { return ep_mask_; }

  latent& set_oo(bool val) {
    if (val ^ oo_) { hash_ ^= zobrist_src_->get_oo(); }
    oo_ = val;
    return *this;
  }

  latent& set_ooo(bool val) {
    if (val ^ ooo_) { hash_ ^= zobrist_src_->get_ooo(); }
    ooo_ = val;
    return *this;
  }

  latent& clear_ep_mask() {
    if (ep_mask_.any()) { hash_ ^= zobrist_src_->get_ep_mask(ep_mask_.item()); }
    ep_mask_ = square_set{};
    return *this;
  }

  template <typename S>
  latent& set_ep_mask(const S& at) {
    static_assert(is_square_v<S>, "at must be of square type");
    clear_ep_mask();
    hash_ ^= zobrist_src_->get_ep_mask(at);
    ep_mask_.add_(at);
    return *this;
  }

  latent(const latent_zobrist_src* src) : zobrist_src_{src} {}
};

struct sided_latent : sided<sided_latent, latent> {
  static inline const latent_zobrist_src w_latent_src{};
  static inline const latent_zobrist_src b_latent_src{};
  static inline const zobrist::hash_type turn_white_src = zobrist::random_bit_string();
  static inline const zobrist::hash_type turn_black_src = zobrist::random_bit_string();

  size_t half_clock{0};
  size_t ply_count{0};
  latent white;
  latent black;

  zobrist::hash_type hash() const {
    const zobrist::hash_type result = white.hash() ^ black.hash();
    return ((ply_count % 2) == 0) ? (result ^ turn_white_src) : (result ^ turn_black_src);
  }

  sided_latent() : white(&w_latent_src), black(&b_latent_src) {}
};

}  // namespace chess
