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

#include <chess/board_history.h>
#include <chess/board_state.h>
#include <chess/move_list.h>
#include <chess/piece_configuration.h>
#include <chess/square.h>
#include <chess/types.h>
#include <feature/util.h>
#include <zobrist/util.h>

#include <cstddef>
#include <iostream>
#include <string>
#include <tuple>

namespace chess {

template <bool noisy_value, bool check_value, bool quiet_value>
struct move_generator_mode {
  static constexpr bool noisy = noisy_value;
  static constexpr bool check = check_value;
  static constexpr bool quiet = quiet_value;
};

namespace generation_mode {

using noisy_and_check = move_generator_mode<true, true, false>;
using quiet_and_check = move_generator_mode<false, true, true>;
using noisy = move_generator_mode<true, false, false>;
using check = move_generator_mode<false, true, false>;
using quiet = move_generator_mode<false, false, true>;
using all = move_generator_mode<true, true, true>;

}  // namespace generation_mode

struct move_generator_info {
  square_set occ;
  square_set last_rank;
  square_set checkers;
  square_set checker_rays;
  square_set pinned;
  square_set king_danger;
  square_set king_diagonal;
  square_set king_horizontal;
};

struct board {
  static constexpr std::size_t num_fen_tokens = 6;

  sided_manifest man_{};
  sided_latent lat_{};

  [[nodiscard]] inline bool turn() const noexcept { return lat_.ply_count % 2 == 0; }
  [[nodiscard]] inline bool is_rule50_draw() const noexcept { return lat_.half_clock >= 100; }
  [[nodiscard]] inline zobrist::hash_type hash() const noexcept { return man_.hash() ^ lat_.hash(); }

  template <color c>
  [[nodiscard]] std::tuple<piece_type, square> least_valuable_attacker(const square& tgt, const square_set& ignore) const noexcept;

  template <color c>
  [[nodiscard]] inline std::tuple<square_set, square_set> checkers(const square_set& occ) const noexcept;

  template <color c>
  [[nodiscard]] inline square_set threat_mask() const noexcept;
  [[nodiscard]] square_set us_threat_mask() const noexcept;
  [[nodiscard]] square_set them_threat_mask() const noexcept;

  template <color c>
  [[nodiscard]] inline bool creates_threat_(const move& mv) const noexcept;
  [[nodiscard]] bool creates_threat(const move& mv) const noexcept;

  template <color c>
  [[nodiscard]] inline square_set king_danger() const noexcept;

  template <color c>
  [[nodiscard]] inline square_set pinned() const noexcept;

  template <color c, typename mode>
  inline void add_en_passant(move_list& mv_ls) const noexcept;

  template <color c, typename mode>
  inline void add_castle(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_normal_pawn(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_normal_knight(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_normal_bishop(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_normal_rook(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_normal_queen(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_pinned_pawn(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_pinned_bishop(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_pinned_rook(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_pinned_queen(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_checked_pawn(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_checked_knight(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_checked_rook(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_checked_bishop(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_checked_queen(const move_generator_info& info, move_list& result) const noexcept;

  template <color c, typename mode>
  inline void add_king(const move_generator_info& info, move_list& result) const noexcept;

  template <color c>
  [[nodiscard]] inline move_generator_info get_move_generator_info() const noexcept;

  template <color c, typename mode>
  [[nodiscard]] inline move_list generate_moves_() const noexcept;

  template <typename mode = generation_mode::all>
  [[nodiscard]] move_list generate_moves() const noexcept;

  template <color c, typename mode>
  [[nodiscard]] inline bool is_legal_(const move& mv) const noexcept;

  template <typename mode>
  [[nodiscard]] bool is_legal(const move& mv) const noexcept;

  template <color c>
  [[nodiscard]] inline bool is_check_() const noexcept;
  [[nodiscard]] bool is_check() const noexcept;

  template <color c, typename T>
  [[nodiscard]] inline bool see_ge_(const move& mv, const T& threshold) const noexcept;

  template <typename T>
  [[nodiscard]] bool see_ge(const move& mv, const T& threshold) const noexcept;

  template <typename T>
  [[nodiscard]] bool see_gt(const move& mv, const T& threshold) const noexcept;

  template <typename T>
  [[nodiscard]] T phase() const noexcept;

  [[nodiscard]] bool has_non_pawn_material() const noexcept;

  template <color c>
  [[nodiscard]] inline bool is_passed_push_(const move& mv) const noexcept;
  [[nodiscard]] bool is_passed_push(const move& mv) const noexcept;

  template <color c>
  [[nodiscard]] std::size_t side_num_pieces() const noexcept;
  [[nodiscard]] std::size_t num_pieces() const noexcept;

  [[nodiscard]] bool is_trivially_drawn() const noexcept;

  template <color c>
  [[nodiscard]] board forward_(const move& mv) const noexcept;
  [[nodiscard]] board forward(const move& mv) const noexcept;

  [[nodiscard]] board mirrored() const noexcept;

  template <typename T>
  void feature_full_reset(T& sided_set) const {
    namespace h_ka = feature::half_ka;

    const square white_king = man_.white.king().item();
    const square black_king = man_.black.king().item();

    sided_set.white.clear();
    sided_set.black.clear();

    over_types([&](const piece_type& pt) {
      for (const auto sq : man_.white.get_plane(pt)) {
        sided_set.white.insert(h_ka::index<color::white, color::white>(white_king, pt, sq));
        sided_set.black.insert(h_ka::index<color::black, color::white>(black_king, pt, sq));
      }
    });

    over_types([&](const piece_type& pt) {
      for (const auto sq : man_.black.get_plane(pt)) {
        sided_set.white.insert(h_ka::index<color::white, color::black>(white_king, pt, sq));
        sided_set.black.insert(h_ka::index<color::black, color::black>(black_king, pt, sq));
      }
    });
  }

  template <color c, typename T0, typename T1>
  void half_feature_partial_reset_(const move& mv, T0& feature_reset_cache, T1& sided_set) const {
    namespace h_ka = feature::half_ka;
    const square our_king = mv.to();

    auto& entry = feature_reset_cache.template us<c>().look_up(our_king);
    sided_piece_configuration& config = entry.config;

    over_types([&](const piece_type& pt) {
      const square_set them_entry_plane = config.them<c>().get_plane(pt);
      const square_set us_entry_plane = config.us<c>().get_plane(pt);

      const square_set them_board_plane = man_.them<c>().get_plane(pt).excluding(mv.to());
      const square_set us_board_plane = [&] {
        if (pt == piece_type::king) { return square_set::of(our_king); }
        return man_.us<c>().get_plane(pt).excluding(mv.from());
      }();

      for (const auto sq : them_entry_plane & ~them_board_plane) { entry.erase(h_ka::index<c, opponent<c>>(our_king, pt, sq)); }
      for (const auto sq : (us_entry_plane & ~us_board_plane)) { entry.erase(h_ka::index<c, c>(our_king, pt, sq)); }

      for (const auto sq : them_board_plane & ~them_entry_plane) { entry.insert(h_ka::index<c, opponent<c>>(our_king, pt, sq)); }
      for (const auto sq : us_board_plane & ~us_entry_plane) { entry.insert(h_ka::index<c, c>(our_king, pt, sq)); }

      config.them<c>().set_plane(pt, them_board_plane);
      config.us<c>().set_plane(pt, us_board_plane);
    });

    entry.copy_state_to(sided_set.template us<c>());
  }

  template <color pov, color p, typename T>
  void half_feature_move_delta_(const move& mv, T& sided_set) const {
    namespace h_ka = feature::half_ka;
    const square our_king = man_.us<pov>().king().item();
    const std::size_t erase_idx_0 = h_ka::index<pov, p>(our_king, mv.piece(), mv.from());

    const std::size_t insert_idx = [&] {
      const piece_type on_to = mv.is_promotion<p>() ? mv.promotion() : mv.piece();
      return h_ka::index<pov, p>(our_king, on_to, mv.to());
    }();

    if (mv.is_capture()) {
      const std::size_t erase_idx_1 = h_ka::index<pov, opponent<p>>(our_king, mv.captured(), mv.to());
      sided_set.template us<pov>().copy_parent_insert_erase_erase(insert_idx, erase_idx_0, erase_idx_1);
      return;
    }

    if (mv.is_enpassant()) {
      const std::size_t erase_idx_1 = h_ka::index<pov, opponent<p>>(our_king, piece_type::pawn, mv.enpassant_sq());
      sided_set.template us<pov>().copy_parent_insert_erase_erase(insert_idx, erase_idx_0, erase_idx_1);
      return;
    }

    sided_set.template us<pov>().copy_parent_insert_erase(insert_idx, erase_idx_0);
  }

  template <color c, typename T0, typename T1>
  void feature_move_delta_(const move& mv, T0& feature_reset_cache, T1& sided_set) const {
    namespace h_ka = feature::half_ka;

    if (mv.is_castle_oo<c>() || mv.is_castle_ooo<c>()) {
      forward_<c>(mv).feature_full_reset(sided_set);
      return;
    }

    if (mv.is_king_move()) {
      half_feature_partial_reset_<c>(mv, feature_reset_cache, sided_set);
      half_feature_move_delta_<opponent<c>, c>(mv, sided_set);
      return;
    }

    half_feature_move_delta_<c, c>(mv, sided_set);
    half_feature_move_delta_<opponent<c>, c>(mv, sided_set);
  }

  template <typename T0, typename T1>
  void feature_move_delta(const move& mv, T0& feature_reset_cache, T1& sided_set) const {
    if (turn()) {
      feature_move_delta_<color::white>(mv, feature_reset_cache, sided_set);
    } else {
      feature_move_delta_<color::black>(mv, feature_reset_cache, sided_set);
    }
  }

  [[nodiscard]] std::tuple<board_history, board> after_uci_moves(const std::string& moves) const noexcept;

  [[nodiscard]] std::string fen() const noexcept;

  [[nodiscard]] static board start_pos() noexcept;

  [[nodiscard]] static board parse_fen(const std::string& fen) noexcept;
};

[[maybe_unused]] std::ostream& operator<<(std::ostream& ostr, const board& bd) noexcept;

}  // namespace chess