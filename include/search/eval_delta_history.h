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

#include <chess/types.h>
#include <search/composite_feature_hash.h>
#include <search/search_constants.h>
#include <search/transposition_table.h>
#include <zobrist/util.h>

#include <cstddef>

namespace search {

struct delta_history_move_zobrist_src {
  static constexpr std::size_t num_squares = 64;
  using plane_t = std::array<zobrist::quarter_hash_type, num_squares>;

  plane_t pawn_{};
  plane_t knight_{};
  plane_t bishop_{};
  plane_t rook_{};
  plane_t queen_{};
  plane_t king_{};

  [[nodiscard]] constexpr std::array<zobrist::quarter_hash_type, num_squares>& get_plane(const chess::piece_type& pt) noexcept {
    return chess::get_member(pt, *this);
  }

  [[nodiscard]] constexpr const std::array<zobrist::quarter_hash_type, num_squares>& get_plane(const chess::piece_type& pt) const noexcept {
    return chess::get_member(pt, *this);
  }

  template <typename S>
  [[nodiscard]] constexpr zobrist::quarter_hash_type get(const chess::piece_type& pt, const S& at) const noexcept {
    static_assert(chess::is_square_v<S>, "at must be of square type");
    return get_plane(pt)[at.index()];
  }

  constexpr delta_history_move_zobrist_src(zobrist::xorshift_generator generator) noexcept {
    chess::over_types([&, this](const chess::piece_type pt) {
      plane_t& pt_plane = get_plane(pt);
      for (auto& elem : pt_plane) { elem = zobrist::lower_quarter(generator.next()); }
    });
  }
};

template <std::size_t N>
struct eval_delta_history {
  static constexpr std::size_t M = 4096;
  static constexpr std::size_t mask = M - 1;
  static_assert((M & mask) == 0);

  static constexpr score_type eval_delta_scale = 256;
  static constexpr delta_history_move_zobrist_src move_zobrist_src{zobrist::xorshift_generator(zobrist::entropy_6)};

  std::array<score_type, M> data{};

  [[nodiscard]] static constexpr std::size_t hash_function(const composite_feature_hash<N>& feature_hash, const chess::move& mv) noexcept {
    const zobrist::quarter_hash_type mv_hash = move_zobrist_src.get(mv.piece(), mv.to());
    return (mv_hash ^ feature_hash.reduced()) & mask;
  }

  [[nodiscard]] constexpr score_type delta_for(const composite_feature_hash<N>& feature_hash, const chess::move& mv) const noexcept {
    const score_type raw_delta = data[hash_function(feature_hash, mv)];
    return raw_delta / eval_delta_scale;
  }

  constexpr void
  update(const composite_feature_hash<N>& feature_hash, const chess::move& mv, const bound_type& bound, const score_type& error) noexcept {
    if (bound == bound_type::upper && error >= 0) { return; }
    if (bound == bound_type::lower && error <= 0) { return; }

    constexpr score_type score_delta_limit = 65536;

    constexpr score_type filter_alpha = 1;
    constexpr score_type filter_c_alpha = 255;
    constexpr score_type filter_divisor = filter_alpha + filter_c_alpha;

    auto& delta = data[hash_function(feature_hash, mv)];

    const score_type scaled_error = error * eval_delta_scale;
    delta = (delta * filter_c_alpha + scaled_error * filter_alpha) / filter_divisor;
    delta = std::clamp(delta, -score_delta_limit, score_delta_limit);
  }

  void clear() noexcept { return data.fill(score_type{}); }
};

constexpr std::size_t eval_delta_history_num_hashes = 2;

struct sided_eval_delta_history : public chess::sided<sided_eval_delta_history, eval_delta_history<eval_delta_history_num_hashes>> {
  using hash_type = composite_feature_hash<eval_delta_history_num_hashes>;
  eval_delta_history<eval_delta_history_num_hashes> white;
  eval_delta_history<eval_delta_history_num_hashes> black;

  void clear() noexcept {
    white.clear();
    black.clear();
  }

  sided_eval_delta_history() : white{}, black{} {}
};

}  // namespace search