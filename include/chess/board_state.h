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
#include <zobrist/util.h>

#include <array>

namespace chess {

struct manifest_zobrist_src {
  static constexpr std::size_t num_squares = 64;
  using plane_t = std::array<zobrist::hash_type, num_squares>;
  plane_t pawn_{};
  plane_t knight_{};
  plane_t bishop_{};
  plane_t rook_{};
  plane_t queen_{};
  plane_t king_{};

  [[nodiscard]] constexpr std::array<zobrist::hash_type, num_squares>& get_plane(const piece_type& pt) noexcept { return get_member(pt, *this); }

  [[nodiscard]] constexpr const std::array<zobrist::hash_type, num_squares>& get_plane(const piece_type& pt) const noexcept {
    return get_member(pt, *this);
  }

  template <typename S>
  [[nodiscard]] zobrist::hash_type get(const piece_type& pt, const S& at) const noexcept;

  manifest_zobrist_src() noexcept;
};

struct manifest {
  const manifest_zobrist_src* zobrist_src_;
  zobrist::hash_type hash_{0};
  square_set pawn_{};
  square_set knight_{};
  square_set bishop_{};
  square_set rook_{};
  square_set queen_{};
  square_set king_{};
  square_set all_{};

  [[nodiscard]] constexpr zobrist::hash_type hash() const noexcept { return hash_; }

  [[nodiscard]] constexpr square_set& get_plane(const piece_type pt) noexcept { return get_member(pt, *this); }

  [[nodiscard]] constexpr piece_type occ(const tbl_square& at) const noexcept { return occ(at.to_square()); }

  [[nodiscard]] constexpr piece_type occ(const square& at) const {
    if (knight_.is_member(at)) { return piece_type::knight; }
    if (bishop_.is_member(at)) { return piece_type::bishop; }
    if (rook_.is_member(at)) { return piece_type::rook; }
    if (queen_.is_member(at)) { return piece_type::queen; }
    if (king_.is_member(at)) { return piece_type::king; }
    return piece_type::pawn;
  }

  [[nodiscard]] constexpr const square_set& all() const noexcept { return all_; }
  [[nodiscard]] constexpr const square_set& pawn() const noexcept { return pawn_; }
  [[nodiscard]] constexpr const square_set& knight() const noexcept { return knight_; }
  [[nodiscard]] constexpr const square_set& bishop() const noexcept { return bishop_; }
  [[nodiscard]] constexpr const square_set& rook() const noexcept { return rook_; }
  [[nodiscard]] constexpr const square_set& queen() const noexcept { return queen_; }
  [[nodiscard]] constexpr const square_set& king() const noexcept { return king_; }

  [[nodiscard]] constexpr const square_set& get_plane(const piece_type pt) const noexcept { return get_member(pt, *this); }

  template <typename S>
  [[maybe_unused]] manifest& add_piece(const piece_type& pt, const S& at) noexcept;

  template <typename S>
  [[maybe_unused]] manifest& remove_piece(const piece_type& pt, const S& at) noexcept;

  manifest(const manifest_zobrist_src* src) noexcept : zobrist_src_{src} {}
};

struct sided_manifest : public sided<sided_manifest, manifest> {
  static inline const manifest_zobrist_src w_manifest_src{};
  static inline const manifest_zobrist_src b_manifest_src{};

  manifest white;
  manifest black;

  [[nodiscard]] constexpr zobrist::hash_type hash() const noexcept { return white.hash() ^ black.hash(); }

  sided_manifest() noexcept : white(&w_manifest_src), black(&b_manifest_src) {}
};

struct latent_zobrist_src {
  static constexpr std::size_t num_squares = 64;
  zobrist::hash_type oo_;
  zobrist::hash_type ooo_;
  std::array<zobrist::hash_type, num_squares> ep_mask_;

  [[nodiscard]] zobrist::hash_type get_oo() const noexcept { return oo_; }
  [[nodiscard]] zobrist::hash_type get_ooo() const noexcept { return ooo_; }

  template <typename S>
  [[nodiscard]] zobrist::hash_type get_ep_mask(const S& at) const noexcept;

  latent_zobrist_src() noexcept;
};

struct latent {
  const latent_zobrist_src* zobrist_src_;
  zobrist::hash_type hash_{0};
  bool oo_{true};
  bool ooo_{true};
  square_set ep_mask_{};

  [[nodiscard]] constexpr const zobrist::hash_type& hash() const noexcept { return hash_; }

  [[nodiscard]] constexpr bool oo() const noexcept { return oo_; }
  [[nodiscard]] constexpr bool ooo() const noexcept { return ooo_; }

  [[nodiscard]] constexpr const square_set& ep_mask() const noexcept { return ep_mask_; }

  [[maybe_unused]] latent& set_oo(const bool val) noexcept;
  [[maybe_unused]] latent& set_ooo(const bool val) noexcept;

  [[maybe_unused]] latent& clear_ep_mask() noexcept;

  template <typename S>
  [[maybe_unused]] latent& set_ep_mask(const S& at) noexcept;

  latent(const latent_zobrist_src* src) noexcept : zobrist_src_{src} {}
};

struct sided_latent : public sided<sided_latent, latent> {
  static inline const latent_zobrist_src w_latent_src{};
  static inline const latent_zobrist_src b_latent_src{};
  static inline const zobrist::hash_type turn_white_src = zobrist::random_bit_string();
  static inline const zobrist::hash_type turn_black_src = zobrist::random_bit_string();

  std::size_t half_clock{0};
  std::size_t ply_count{0};
  latent white;
  latent black;

  [[nodiscard]] inline zobrist::hash_type hash() const noexcept {
    const zobrist::hash_type result = white.hash() ^ black.hash();
    return ((ply_count % 2) == 0) ? (result ^ turn_white_src) : (result ^ turn_black_src);
  }

  sided_latent() noexcept : white(&w_latent_src), black(&b_latent_src) {}
};

}  // namespace chess
