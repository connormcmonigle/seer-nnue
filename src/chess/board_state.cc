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

#include <chess/board_state.h>

#include <algorithm>

namespace chess {

template <typename S>
zobrist::hash_type manifest_zobrist_src::get(const piece_type& pt, const S& at) const noexcept {
  static_assert(is_square_v<S>, "at must be of square type");
  return get_plane(pt)[at.index()];
}

manifest_zobrist_src::manifest_zobrist_src() noexcept {
  over_types([this](const piece_type pt) {
    plane_t& pt_plane = get_plane(pt);
    std::transform(pt_plane.begin(), pt_plane.end(), pt_plane.begin(), [](auto&&...) { return zobrist::random_bit_string(); });
  });
}

template <typename S>
manifest& manifest::add_piece(const piece_type& pt, const S& at) noexcept {
  static_assert(is_square_v<S>, "at must be of square type");
  if (pt == piece_type::pawn) { pawn_hash_ ^= zobrist_src_->get(pt, at); }
  hash_ ^= zobrist_src_->get(pt, at);
  all_ |= at.bit_board();
  get_plane(pt) |= at.bit_board();
  return *this;
}

template <typename S>
manifest& manifest::remove_piece(const piece_type& pt, const S& at) noexcept {
  static_assert(is_square_v<S>, "at must be of square type");
  if (pt == piece_type::pawn) { pawn_hash_ ^= zobrist_src_->get(pt, at); }
  hash_ ^= zobrist_src_->get(pt, at);
  all_ &= ~at.bit_board();
  get_plane(pt) &= ~at.bit_board();
  return *this;
}

template <typename S>
zobrist::hash_type latent_zobrist_src::get_ep_mask(const S& at) const noexcept {
  static_assert(is_square_v<S>, "at must be of square type");
  return ep_mask_[at.index()];
}

latent_zobrist_src::latent_zobrist_src() noexcept {
  oo_ = zobrist::random_bit_string();
  ooo_ = zobrist::random_bit_string();
  std::transform(ep_mask_.begin(), ep_mask_.end(), ep_mask_.begin(), [](auto&&...) { return zobrist::random_bit_string(); });
}

latent& latent::set_oo(const bool val) noexcept {
  if (val ^ oo_) { hash_ ^= zobrist_src_->get_oo(); }
  oo_ = val;
  return *this;
}

latent& latent::set_ooo(const bool val) noexcept {
  if (val ^ ooo_) { hash_ ^= zobrist_src_->get_ooo(); }
  ooo_ = val;
  return *this;
}

latent& latent::clear_ep_mask() noexcept {
  if (ep_mask_.any()) { hash_ ^= zobrist_src_->get_ep_mask(ep_mask_.item()); }
  ep_mask_ = square_set{};
  return *this;
}

template <typename S>
latent& latent::set_ep_mask(const S& at) noexcept {
  static_assert(is_square_v<S>, "at must be of square type");
  clear_ep_mask();
  hash_ ^= zobrist_src_->get_ep_mask(at);
  ep_mask_.insert(at);
  return *this;
}

}  // namespace chess

template zobrist::hash_type chess::manifest_zobrist_src::get(const chess::piece_type&, const chess::tbl_square&) const noexcept;
template zobrist::hash_type chess::manifest_zobrist_src::get(const chess::piece_type&, const chess::square&) const noexcept;

template chess::manifest& chess::manifest::add_piece(const piece_type&, const chess::tbl_square&) noexcept;
template chess::manifest& chess::manifest::add_piece(const piece_type&, const chess::square&) noexcept;

template chess::manifest& chess::manifest::remove_piece(const piece_type&, const chess::tbl_square&) noexcept;
template chess::manifest& chess::manifest::remove_piece(const piece_type&, const chess::square&) noexcept;

template zobrist::hash_type chess::latent_zobrist_src::get_ep_mask(const chess::tbl_square&) const noexcept;
template zobrist::hash_type chess::latent_zobrist_src::get_ep_mask(const chess::square&) const noexcept;

template chess::latent& chess::latent::set_ep_mask(const chess::tbl_square&) noexcept;
template chess::latent& chess::latent::set_ep_mask(const chess::square&) noexcept;
