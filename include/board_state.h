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
#include <move.h>
#include <square.h>
#include <table_generation.h>
#include <zobrist_util.h>

#include <array>
#include <limits>
#include <tuple>
#include <vector>

namespace chess {

struct manifest {
  static constexpr size_t num_squares = 64;

  const zobrist::manifest_src* zobrist_src_;
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

  manifest(const zobrist::manifest_src* src) : zobrist_src_{src} {}
};

struct sided_manifest : sided<sided_manifest, manifest> {
  manifest white;
  manifest black;

  zobrist::hash_type hash() const { return white.hash() ^ black.hash(); }

  sided_manifest() : white(&zobrist::sources::manifest.white), black(&zobrist::sources::manifest.black) {}
};

struct latent {
  const zobrist::latent_src* zobrist_src_;
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
    ep_mask_.insert(at);
    return *this;
  }

  latent(const zobrist::latent_src* src) : zobrist_src_{src} {}
};

struct sided_latent : sided<sided_latent, latent> {
  size_t half_clock{0};
  size_t ply_count{0};
  latent white;
  latent black;

  zobrist::hash_type hash() const {
    const zobrist::hash_type result = white.hash() ^ black.hash();
    return result ^ zobrist::sources::turn.us(ply_count % 2 == 0);
  }

  sided_latent() : white(&zobrist::sources::latent.white), black(&zobrist::sources::latent.black) {}
};

}  // namespace chess
