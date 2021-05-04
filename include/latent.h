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

#include <array>
#include <limits>
#include <algorithm>
#include <tuple>

#include <zobrist_util.h>
#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>

namespace chess {

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
    std::transform(ep_mask_.begin(), ep_mask_.end(), ep_mask_.begin(),
                   [](auto...) { return zobrist::random_bit_string(); });
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
    if (val ^ oo_) {
      hash_ ^= zobrist_src_->get_oo();
    }
    oo_ = val;
    return *this;
  }

  latent& set_ooo(bool val) {
    if (val ^ ooo_) {
      hash_ ^= zobrist_src_->get_ooo();
    }
    ooo_ = val;
    return *this;
  }

  latent& clear_ep_mask() {
    if (ep_mask_.any()) {
      hash_ ^= zobrist_src_->get_ep_mask(ep_mask_.item());
    }
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
  static inline const zobrist::hash_type turn_white_src =
      zobrist::random_bit_string();
  static inline const zobrist::hash_type turn_black_src =
      zobrist::random_bit_string();

  size_t half_clock{0};
  size_t move_count{0};
  latent white;
  latent black;

  zobrist::hash_type hash() const {
    const zobrist::hash_type result = white.hash() ^ black.hash();
    return ((move_count % 2) == 0) ? (result ^ turn_white_src)
                                   : (result ^ turn_black_src);
  }

  sided_latent() : white(&w_latent_src), black(&b_latent_src) {}
};
}
