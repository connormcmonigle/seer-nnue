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

#include <board.h>
#include <nnue_model.h>
#include <nnue_util.h>
#include <search_constants.h>
#include <zobrist_util.h>

#include <array>
#include <optional>

namespace nnue {

struct piece_configuration {
  chess::square_set pawn_{};
  chess::square_set knight_{};
  chess::square_set bishop_{};
  chess::square_set rook_{};
  chess::square_set queen_{};
  chess::square_set king_{};

  chess::square_set& get_plane(const chess::piece_type& pt) { return chess::get_member(pt, *this); }
  const chess::square_set& get_plane(const chess::piece_type& pt) const { return chess::get_member(pt, *this); }
};

struct sided_piece_configuration : chess::sided<sided_piece_configuration, piece_configuration> {
  piece_configuration white;
  piece_configuration black;

  sided_piece_configuration() : white{}, black{} {}
};

struct feature_reset_cache_entry {
  static constexpr size_t dim = weights::base_dim;

  using parameter_type = weights::quantized_parameter_type;
  using weights_type = big_affine<parameter_type, feature::half_ka::numel, dim>;

  const weights_type* weights_;
  aligned_slice<parameter_type, dim> slice_;
  sided_piece_configuration config;

  void insert(const size_t& idx) { weights_->insert_idx(idx, slice_); }
  void erase(const size_t& idx) { weights_->erase_idx(idx, slice_); }
  void copy_state_to(feature_transformer<parameter_type>& dst) const { dst.slice_.copy_from(slice_); }

  void reinitialize(const weights_type* weights, const aligned_slice<parameter_type, dim>& slice) {
    weights_ = weights;
    slice_ = slice;
    config = sided_piece_configuration{};

    slice_.copy_from(weights_->b);
  }

  feature_reset_cache_entry() : slice_{nullptr}, config{} {}
};

struct feature_reset_cache {
  using entry_type = feature_reset_cache_entry;
  static constexpr size_t num_squares = 64;

  stack_scratchpad<entry_type::parameter_type, num_squares * entry_type::dim> scratchpad_{};
  feature_reset_cache_entry entries_[num_squares]{};

  feature_reset_cache_entry* look_up(const chess::square& sq) { return entries_ + sq.index(); }

  void reinitialize(const weights* weights) {
    for (size_t i(0); i < num_squares; ++i) {
      const auto slice = scratchpad_.get_nth_slice<entry_type::dim>(i);
      entries_[i].reinitialize(&weights->quantized_shared, slice);
    }
  }
};

struct sided_feature_reset_cache : chess::sided<sided_feature_reset_cache, feature_reset_cache> {
  feature_reset_cache white;
  feature_reset_cache black;

  void reinitialize(const weights* weights) {
    white.reinitialize(weights);
    black.reinitialize(weights);
  }

  sided_feature_reset_cache() : white{}, black{} {}
};

}  // namespace nnue