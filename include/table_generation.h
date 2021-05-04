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

#include <iostream>
#include <cmath>
#include <cassert>
#include <array>
#include <vector>
#include <string>
#include <string_view>
#include <sstream>
#include <type_traits>
#include <algorithm>

#include <enum_util.h>
#include <square.h>

namespace chess {

constexpr std::array<std::uint64_t, 64> rook_magics = {
    0xA180022080400230ull, 0x0040100040022000ull, 0x0080088020001002ull,
    0x0080080280841000ull, 0x4200042010460008ull, 0x04800A0003040080ull,
    0x0400110082041008ull, 0x008000A041000880ull, 0x10138001A080C010ull,
    0x0000804008200480ull, 0x00010011012000C0ull, 0x0022004128102200ull,
    0x000200081201200Cull, 0x202A001048460004ull, 0x0081000100420004ull,
    0x4000800380004500ull, 0x0000208002904001ull, 0x0090004040026008ull,
    0x0208808010002001ull, 0x2002020020704940ull, 0x8048010008110005ull,
    0x6820808004002200ull, 0x0A80040008023011ull, 0x00B1460000811044ull,
    0x4204400080008EA0ull, 0xB002400180200184ull, 0x2020200080100380ull,
    0x0010080080100080ull, 0x2204080080800400ull, 0x0000A40080360080ull,
    0x02040604002810B1ull, 0x008C218600004104ull, 0x8180004000402000ull,
    0x488C402000401001ull, 0x4018A00080801004ull, 0x1230002105001008ull,
    0x8904800800800400ull, 0x0042000C42003810ull, 0x008408110400B012ull,
    0x0018086182000401ull, 0x2240088020C28000ull, 0x001001201040C004ull,
    0x0A02008010420020ull, 0x0010003009010060ull, 0x0004008008008014ull,
    0x0080020004008080ull, 0x0282020001008080ull, 0x50000181204A0004ull,
    0x48FFFE99FECFAA00ull, 0x48FFFE99FECFAA00ull, 0x497FFFADFF9C2E00ull,
    0x613FFFDDFFCE9200ull, 0xFFFFFFE9FFE7CE00ull, 0xFFFFFFF5FFF3E600ull,
    0x0010301802830400ull, 0x510FFFF5F63C96A0ull, 0xEBFFFFB9FF9FC526ull,
    0x61FFFEDDFEEDAEAEull, 0x53BFFFEDFFDEB1A2ull, 0x127FFFB9FFDFB5F6ull,
    0x411FFFDDFFDBF4D6ull, 0x0801000804000603ull, 0x0003FFEF27EEBE74ull,
    0x7645FFFECBFEA79Eull,
};

constexpr std::array<std::uint64_t, 64> bishop_magics = {
    0xFFEDF9FD7CFCFFFFull, 0xFC0962854A77F576ull, 0x5822022042000000ull,
    0x2CA804A100200020ull, 0x0204042200000900ull, 0x2002121024000002ull,
    0xFC0A66C64A7EF576ull, 0x7FFDFDFCBD79FFFFull, 0xFC0846A64A34FFF6ull,
    0xFC087A874A3CF7F6ull, 0x1001080204002100ull, 0x1810080489021800ull,
    0x0062040420010A00ull, 0x5028043004300020ull, 0xFC0864AE59B4FF76ull,
    0x3C0860AF4B35FF76ull, 0x73C01AF56CF4CFFBull, 0x41A01CFAD64AAFFCull,
    0x040C0422080A0598ull, 0x4228020082004050ull, 0x0200800400E00100ull,
    0x020B001230021040ull, 0x7C0C028F5B34FF76ull, 0xFC0A028E5AB4DF76ull,
    0x0020208050A42180ull, 0x001004804B280200ull, 0x2048020024040010ull,
    0x0102C04004010200ull, 0x020408204C002010ull, 0x02411100020080C1ull,
    0x102A008084042100ull, 0x0941030000A09846ull, 0x0244100800400200ull,
    0x4000901010080696ull, 0x0000280404180020ull, 0x0800042008240100ull,
    0x0220008400088020ull, 0x04020182000904C9ull, 0x0023010400020600ull,
    0x0041040020110302ull, 0xDCEFD9B54BFCC09Full, 0xF95FFA765AFD602Bull,
    0x1401210240484800ull, 0x0022244208010080ull, 0x1105040104000210ull,
    0x2040088800C40081ull, 0x43FF9A5CF4CA0C01ull, 0x4BFFCD8E7C587601ull,
    0xFC0FF2865334F576ull, 0xFC0BF6CE5924F576ull, 0x80000B0401040402ull,
    0x0020004821880A00ull, 0x8200002022440100ull, 0x0009431801010068ull,
    0xC3FFB7DC36CA8C89ull, 0xC3FF8A54F4CA2C89ull, 0xFFFFFCFCFD79EDFFull,
    0xFC0863FCCB147576ull, 0x040C000022013020ull, 0x2000104000420600ull,
    0x0400000260142410ull, 0x0800633408100500ull, 0xFC087E8E4BB2F736ull,
    0x43FF9E4EF4CA2C89ull,
};

constexpr std::uint64_t deposit(std::uint64_t src, std::uint64_t mask) {
  std::uint64_t res{0};
  for (std::uint64_t bb{1}; mask; bb += bb) {
    if (src & bb) {
      res |= mask & -mask;
    }
    mask &= mask - static_cast<uint64_t>(1);
  }
  return res;
}

constexpr std::array<delta, 8> queen_deltas() {
  using D = delta;
  return {D{1, 0},  D{0, 1},  D{-1, 0},  D{0, -1},
          D{1, -1}, D{-1, 1}, D{-1, -1}, D{1, 1}};
}

constexpr std::array<delta, 8> king_deltas() {
  using D = delta;
  return {D{1, 0},  D{0, 1},  D{-1, 0},  D{0, -1},
          D{1, -1}, D{-1, 1}, D{-1, -1}, D{1, 1}};
}

constexpr std::array<delta, 8> knight_deltas() {
  using D = delta;
  return {D{1, 2},  D{2, 1},  D{-1, 2},  D{2, -1},
          D{1, -2}, D{-2, 1}, D{-1, -2}, D{-2, -1}};
}

constexpr std::array<delta, 4> bishop_deltas() {
  using D = delta;
  return {D{1, -1}, D{-1, 1}, D{-1, -1}, D{1, 1}};
}

constexpr std::array<delta, 4> rook_deltas() {
  using D = delta;
  return {D{1, 0}, D{0, 1}, D{-1, 0}, D{0, -1}};
}

template <color C>
struct pawn_delta {};

template <>
struct pawn_delta<color::white> {
  static constexpr int start_rank_idx = 1;
  static constexpr int last_rank_idx = 7;
  static constexpr int double_rank_idx = 3;
  static constexpr square_set start_rank = gen_rank(start_rank_idx);
  static constexpr square_set last_rank = gen_rank(last_rank_idx);
  static constexpr square_set double_rank = gen_rank(double_rank_idx);
  static constexpr std::array<delta, 2> attack = {delta{-1, 1}, delta{1, 1}};
  static constexpr delta step = delta{0, 1};
};

template <>
struct pawn_delta<color::black> {
  static constexpr int start_rank_idx = 6;
  static constexpr int last_rank_idx = 0;
  static constexpr int double_rank_idx = 4;
  static constexpr square_set start_rank = gen_rank(start_rank_idx);
  static constexpr square_set last_rank = gen_rank(last_rank_idx);
  static constexpr square_set double_rank = gen_rank(double_rank_idx);
  static constexpr std::array<delta, 2> attack = {delta{-1, -1}, delta{1, -1}};
  static constexpr delta step = delta{0, -1};
};

template <typename F, typename D>
constexpr void over_all_step_attacks(const D& deltas, F&& f) {
  auto do_shift = [f](const tbl_square from, const delta d) {
    if (auto to = from.add(d); to.is_valid()) {
      f(from, to);
    }
  };
  over_all([&](const tbl_square from) {
    for (const delta d : deltas) {
      do_shift(from, d);
    }
  });
}

template <color C>
struct castle_info_ {};

template <>
struct castle_info_<color::white> {
  static constexpr tbl_square oo_rook_tbl{0, 0};
  static constexpr tbl_square ooo_rook_tbl{7, 0};
  static constexpr tbl_square start_king_tbl{3, 0};

  static constexpr tbl_square after_oo_rook_tbl{2, 0};
  static constexpr tbl_square after_ooo_rook_tbl{4, 0};
  static constexpr tbl_square after_oo_king_tbl{1, 0};
  static constexpr tbl_square after_ooo_king_tbl{5, 0};

  square oo_rook;
  square ooo_rook;
  square start_king;

  square after_oo_rook;
  square after_ooo_rook;
  square after_oo_king;
  square after_ooo_king;

  square_set oo_mask;

  square_set ooo_danger_mask;
  square_set ooo_occ_mask;

  constexpr castle_info_()
      : oo_rook{oo_rook_tbl.to_square()},
        ooo_rook{ooo_rook_tbl.to_square()},
        start_king{start_king_tbl.to_square()},
        after_oo_rook{after_oo_rook_tbl.to_square()},
        after_ooo_rook{after_ooo_rook_tbl.to_square()},
        after_oo_king{after_oo_king_tbl.to_square()},
        after_ooo_king{after_ooo_king_tbl.to_square()} {
    constexpr delta ooo_delta{1, 0};
    constexpr delta oo_delta{-1, 0};
    for (auto sq = start_king_tbl.add(oo_delta); true; sq = sq.add(oo_delta)) {
      oo_mask.add_(sq);
      if (sq == after_oo_king_tbl) {
        break;
      }
    }
    for (auto sq = start_king_tbl.add(ooo_delta); true;
         sq = sq.add(ooo_delta)) {
      ooo_danger_mask.add_(sq);
      if (sq == after_ooo_king_tbl) {
        break;
      }
    }
    for (auto sq = start_king_tbl.add(ooo_delta); sq != ooo_rook_tbl;
         sq = sq.add(ooo_delta)) {
      ooo_occ_mask.add_(sq);
    }
  }
};

template <>
struct castle_info_<color::black> {
  static constexpr tbl_square oo_rook_tbl{0, 7};
  static constexpr tbl_square ooo_rook_tbl{7, 7};
  static constexpr tbl_square start_king_tbl{3, 7};

  static constexpr tbl_square after_oo_rook_tbl{2, 7};
  static constexpr tbl_square after_ooo_rook_tbl{4, 7};
  static constexpr tbl_square after_oo_king_tbl{1, 7};
  static constexpr tbl_square after_ooo_king_tbl{5, 7};

  square oo_rook;
  square ooo_rook;
  square start_king;

  square after_oo_rook;
  square after_ooo_rook;
  square after_oo_king;
  square after_ooo_king;

  square_set oo_mask;

  square_set ooo_danger_mask;
  square_set ooo_occ_mask;

  constexpr castle_info_()
      : oo_rook{oo_rook_tbl.to_square()},
        ooo_rook{ooo_rook_tbl.to_square()},
        start_king{start_king_tbl.to_square()},
        after_oo_rook{after_oo_rook_tbl.to_square()},
        after_ooo_rook{after_ooo_rook_tbl.to_square()},
        after_oo_king{after_oo_king_tbl.to_square()},
        after_ooo_king{after_ooo_king_tbl.to_square()} {
    constexpr delta ooo_delta{1, 0};
    constexpr delta oo_delta{-1, 0};
    for (auto sq = start_king_tbl.add(oo_delta); true; sq = sq.add(oo_delta)) {
      oo_mask.add_(sq);
      if (sq == after_oo_king_tbl) {
        break;
      }
    }
    for (auto sq = start_king_tbl.add(ooo_delta); true;
         sq = sq.add(ooo_delta)) {
      ooo_danger_mask.add_(sq);
      if (sq == after_ooo_king_tbl) {
        break;
      }
    }
    for (auto sq = start_king_tbl.add(ooo_delta); sq != ooo_rook_tbl;
         sq = sq.add(ooo_delta)) {
      ooo_occ_mask.add_(sq);
    }
  }
};

struct stepper_attack_tbl {
  static constexpr size_t num_squares = 64;
  piece_type type;
  std::array<square_set, num_squares> data{};

  template <typename T>
  constexpr const square_set& look_up(const T& sq) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return data[sq.index()];
  }

  template <typename D>
  constexpr stepper_attack_tbl(piece_type pt, const D& deltas)
      : type{pt} {
    over_all_step_attacks(deltas,
                          [this](const tbl_square& from, const tbl_square& to) {
                            data[from.index()].add_(to);
                          });
  }
};

template <color c>
struct passer_tbl_ {
  static constexpr size_t num_squares = 64;
  std::array<square_set, num_squares> data{};

  template <typename T>
  constexpr square_set mask(const T& sq) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return data[sq.index()];
  }

  constexpr passer_tbl_() {
    over_all([this](const tbl_square& sq) {
      for (auto left = sq.add(pawn_delta<c>::attack[0]); left.is_valid();
           left = left.add(pawn_delta<c>::step)) {
        data[sq.index()].add_(left.to_square());
      }
      for (auto center = sq.add(pawn_delta<c>::step); center.is_valid();
           center = center.add(pawn_delta<c>::step)) {
        data[sq.index()].add_(center.to_square());
      }
      for (auto right = sq.add(pawn_delta<c>::attack[1]); right.is_valid();
           right = right.add(pawn_delta<c>::step)) {
        data[sq.index()].add_(right.to_square());
      }
    });
  }
};

template <color c>
struct pawn_push_tbl_ {
  static constexpr size_t num_squares = 64;

  piece_type type{piece_type::pawn};
  std::array<square_set, num_squares> data{};

  template <typename T>
  constexpr square_set look_up(const T& sq, const square_set& occ) const {
    static_assert(is_square_v<T>, "can only look up squares");
    square_set occ_ = occ;
    if
      constexpr(c == color::white) {
        occ_.data |= (occ_.data & ~sq.bit_board())
                     << static_cast<std::uint64_t>(8);
      }
    else {
      occ_.data |=
          (occ_.data & ~sq.bit_board()) >> static_cast<std::uint64_t>(8);
    }
    return data[sq.index()] & (~occ_);
  }

  constexpr pawn_push_tbl_() {
    over_all([this](const tbl_square& from) {
      if (const tbl_square to = from.add(pawn_delta<c>::step); to.is_valid()) {
        data[from.index()].add_(to);
      }
    });
    over_rank(pawn_delta<c>::start_rank_idx, [this](const tbl_square& from) {
      const tbl_square to =
          from.add(pawn_delta<c>::step).add(pawn_delta<c>::step);
      if (to.is_valid()) {
        data[from.index()].add_(to);
      }
    });
  }
};

template <typename F, typename D>
constexpr void over_all_slide_masks(const D& deltas, F&& f) {
  auto do_ray = [f](const tbl_square from, const delta d) {
    for (auto to = from.add(d); to.add(d).is_valid(); to = to.add(d)) {
      f(from, to);
    }
  };

  over_all([&](const tbl_square from) {
    for (const delta d : deltas) {
      do_ray(from, d);
    }
  });
}

struct slider_mask_tbl {
  static constexpr size_t num_squares = 64;
  piece_type type;
  std::array<square_set, num_squares> data{};

  template <typename T>
  constexpr const square_set& look_up(const T& sq) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return data[sq.index()];
  }

  template <typename D>
  constexpr slider_mask_tbl(const piece_type pt, const D& deltas)
      : type{pt} {
    over_all_slide_masks(deltas,
                         [this](const tbl_square& from, const tbl_square& to) {
                           data[from.index()].add_(to);
                         });
  }
};

template <size_t max_num_blockers>
struct slider_attack_tbl {
  static constexpr size_t minor = 64;
  static constexpr size_t major = static_cast<size_t>(1) << max_num_blockers;
  static constexpr size_t fixed_shift = 64 - max_num_blockers;
  static constexpr std::uint64_t one = static_cast<std::uint64_t>(1);

  piece_type type;
  slider_mask_tbl mask_tbl;
  std::array<std::uint64_t, 64> magics;
  std::array<square_set, minor * major> data{};

  template <typename T>
  constexpr size_t index_offset(const T& sq, const square_set& blocker) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return (magics[sq.index()] * blocker.data) >> fixed_shift;
  }

  template <typename T>
  constexpr square_set look_up(const T& sq, const square_set& blocker) const {
    static_assert(is_square_v<T>, "can only look up squares");
    const square_set mask = mask_tbl.look_up(sq);
    return data[sq.index() * major + index_offset(sq, blocker & mask)];
  }

  template <typename D>
  constexpr square_set compute_rays(const tbl_square& from,
                                    const square_set& blocker,
                                    const D& deltas) const {
    square_set result{};
    for (delta d : deltas) {
      for (auto to = from.add(d); to.is_valid(); to = to.add(d)) {
        result.add_(to);
        if (blocker.occ(to.index())) {
          break;
        }
      }
    }
    return result;
  }

  template <typename D>
  constexpr slider_attack_tbl(const piece_type pt, const D& deltas,
                              const std::array<std::uint64_t, 64>& magic_tbl)
      : type{pt}, mask_tbl(pt, deltas), magics{magic_tbl} {
    over_all([&, this](const tbl_square from) {
      const square_set mask = mask_tbl.look_up(from);
      const std::uint64_t max_blocker = one << pop_count(mask.data);
      for (std::uint64_t blocker_data(0); blocker_data < max_blocker;
           ++blocker_data) {
        const square_set blocker(deposit(blocker_data, mask.data));
        data[major * from.index() + index_offset(from, blocker)] =
            compute_rays(from, blocker, deltas);
      }
    });
  }
};

template <color c>
inline constexpr castle_info_<c> castle_info = castle_info_<c>{};

template <color c>
inline constexpr pawn_push_tbl_<c> pawn_push_tbl = pawn_push_tbl_<c>{};

template <color c>
inline constexpr stepper_attack_tbl pawn_attack_tbl =
    stepper_attack_tbl{piece_type::pawn, pawn_delta<c>::attack};

template <color c>
inline constexpr passer_tbl_<c> passer_tbl = passer_tbl_<c>{};

inline constexpr stepper_attack_tbl knight_attack_tbl{piece_type::knight,
                                                      knight_deltas()};
inline constexpr stepper_attack_tbl king_attack_tbl{piece_type::king,
                                                    king_deltas()};
inline constexpr slider_attack_tbl<9> bishop_attack_tbl{
    piece_type::bishop, bishop_deltas(), bishop_magics};

inline constexpr slider_attack_tbl<12> rook_attack_tbl{
    piece_type::rook, rook_deltas(), rook_magics};
}
