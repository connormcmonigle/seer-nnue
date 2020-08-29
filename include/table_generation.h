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

#include <immintrin.h>

#include <enum_util.h>
#include <square.h>


namespace chess{

constexpr std::uint64_t pext_compile_time(std::uint64_t src, std::uint64_t mask){
  std::uint64_t res{0};
  for(std::uint64_t bb{1}; mask; bb += bb) {
    if(src & (mask & -mask)){
      res |= bb;
    }
    mask &= mask - static_cast<uint64_t>(1);
  }
  return res;
}

inline std::uint64_t pext(std::uint64_t src, std::uint64_t mask){
  #ifdef __BMI__
  return _pext_u64(src, mask);
  #else
  return pext_compile_time(src, mask);
  #endif
}

constexpr std::uint64_t pdep_compile_time(std::uint64_t src, std::uint64_t mask){
  std::uint64_t res{0};
  for(std::uint64_t bb{1}; mask; bb += bb) {
    if(src & bb){
      res |= mask & -mask;
    }
    mask &= mask - static_cast<uint64_t>(1);
  }
  return res;
}

inline std::uint64_t pdep(std::uint64_t src, std::uint64_t mask){
  #ifdef __BMI__
  return _pdep_u64(src, mask);
  #else
  return pdep_compile_time(src, mask);
  #endif
}

constexpr std::array<delta, 8> queen_deltas(){
  using D = delta;
  return {D{1, 0}, D{0, 1}, D{-1, 0}, D{0, -1}, D{1, -1}, D{-1, 1}, D{-1, -1}, D{1, 1}};
}

constexpr std::array<delta, 8> king_deltas(){
  using D = delta;
  return {D{1, 0}, D{0, 1}, D{-1, 0}, D{0, -1}, D{1, -1}, D{-1, 1}, D{-1, -1}, D{1, 1}};
}

constexpr std::array<delta, 8> knight_deltas(){
  using D = delta;
  return {D{1, 2}, D{2, 1}, D{-1, 2}, D{2, -1}, D{1, -2}, D{-2, 1}, D{-1, -2}, D{-2, -1}};
}

constexpr std::array<delta, 4> bishop_deltas(){
  using D = delta;
  return {D{1, -1}, D{-1, 1}, D{-1, -1}, D{1, 1}};
}

constexpr std::array<delta, 4> rook_deltas(){
  using D = delta;
  return {D{1, 0}, D{0, 1}, D{-1, 0}, D{0, -1}};
}

template<color C> struct pawn_delta{};

template<>
struct pawn_delta<color::white>{
  static constexpr int start_rank_idx = 1;
  static constexpr int last_rank_idx = 7;
  static constexpr int double_rank_idx = 3;
  static constexpr square_set start_rank = gen_rank(start_rank_idx);
  static constexpr square_set last_rank = gen_rank(last_rank_idx);
  static constexpr square_set double_rank = gen_rank(double_rank_idx);
  static constexpr std::array<delta, 2> attack = {delta{-1, 1}, delta{1, 1}};
  static constexpr delta step = delta{0, 1};
};

template<>
struct pawn_delta<color::black>{
  static constexpr int start_rank_idx = 6;
  static constexpr int last_rank_idx = 0;
  static constexpr int double_rank_idx = 4;
  static constexpr square_set start_rank = gen_rank(start_rank_idx);
  static constexpr square_set last_rank = gen_rank(last_rank_idx);
  static constexpr square_set double_rank = gen_rank(double_rank_idx);
  static constexpr std::array<delta, 2> attack = {delta{-1, -1}, delta{1, -1}};
  static constexpr delta step = delta{0, -1};
};

template<typename F, typename D>
constexpr void over_all_step_attacks(const D& deltas, F&& f){
  auto do_shift = [f](const tbl_square from, const delta d){
    if(auto to = from.add(d); to.is_valid()){
      f(from, to);
    }
  };
  over_all([&](const tbl_square from){
    for(const delta d : deltas){ do_shift(from, d); }
  });
}

template<color C> struct castle_info_{};

template<>
struct castle_info_<color::white>{
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
  
  constexpr castle_info_() :
    oo_rook{oo_rook_tbl.to_square()},
    ooo_rook{ooo_rook_tbl.to_square()},
    start_king{start_king_tbl.to_square()},
    after_oo_rook{after_oo_rook_tbl.to_square()},
    after_ooo_rook{after_ooo_rook_tbl.to_square()},
    after_oo_king{after_oo_king_tbl.to_square()},
    after_ooo_king{after_ooo_king_tbl.to_square()}
  {
    constexpr delta ooo_delta{1, 0};
    constexpr delta oo_delta{-1, 0};
    for(auto sq = start_king_tbl.add(oo_delta); true; sq = sq.add(oo_delta)){
      oo_mask.add_(sq);
      if(sq == after_oo_king_tbl){ break; }
    }
    for(auto sq = start_king_tbl.add(ooo_delta); true; sq = sq.add(ooo_delta)){
      ooo_danger_mask.add_(sq);
      if(sq == after_ooo_king_tbl){ break; }
    }
    for(auto sq = start_king_tbl.add(ooo_delta); sq != ooo_rook_tbl; sq = sq.add(ooo_delta)){
      ooo_occ_mask.add_(sq);
    }
  }
};

template<>
struct castle_info_<color::black>{
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
  square_set ooo_occ_mask;;
  
  constexpr castle_info_() :
    oo_rook{oo_rook_tbl.to_square()},
    ooo_rook{ooo_rook_tbl.to_square()},
    start_king{start_king_tbl.to_square()},
    after_oo_rook{after_oo_rook_tbl.to_square()},
    after_ooo_rook{after_ooo_rook_tbl.to_square()},
    after_oo_king{after_oo_king_tbl.to_square()},
    after_ooo_king{after_ooo_king_tbl.to_square()}
  {
    constexpr delta ooo_delta{1, 0};
    constexpr delta oo_delta{-1, 0};
    for(auto sq = start_king_tbl.add(oo_delta); true; sq = sq.add(oo_delta)){
      oo_mask.add_(sq);
      if(sq == after_oo_king_tbl){ break; }
    }
    for(auto sq = start_king_tbl.add(ooo_delta); true; sq = sq.add(ooo_delta)){
      ooo_danger_mask.add_(sq);
      if(sq == after_ooo_king_tbl){ break; }
    }
    for(auto sq = start_king_tbl.add(ooo_delta); sq != ooo_rook_tbl; sq = sq.add(ooo_delta)){
      ooo_occ_mask.add_(sq);
    }
  }
};

struct stepper_attack_tbl{
  static constexpr size_t num_squares = 64;
  piece_type type;
  std::array<square_set, num_squares> data{};

  template<typename T>
  constexpr const square_set& look_up(const T& sq) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return data[sq.index()];
  }

  template<typename D>
  constexpr stepper_attack_tbl(piece_type pt, const D& deltas) : type{pt} {
    over_all_step_attacks(deltas, [this](const tbl_square& from, const tbl_square& to){
      data[from.index()].add_(to);
    });
  }
};

template<color c>
struct passer_tbl_{
  static constexpr size_t num_squares = 64;
  std::array<square_set, num_squares> data{};

  template<typename T>
  constexpr square_set mask(const T& sq) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return data[sq.index()];
  }

  constexpr passer_tbl_(){
    over_all([this](const tbl_square& sq){
      for(auto left = sq.add(pawn_delta<c>::attack[0]); left.is_valid(); left = left.add(pawn_delta<c>::step)){
        data[sq.index()].add_(left.to_square());
      }
      for(auto center = sq.add(pawn_delta<c>::step); center.is_valid(); center = center.add(pawn_delta<c>::step)){
        data[sq.index()].add_(center.to_square());
      }
      for(auto right = sq.add(pawn_delta<c>::attack[1]); right.is_valid(); right = right.add(pawn_delta<c>::step)){
        data[sq.index()].add_(right.to_square());
      }
    });
  }
};

template<color c>
struct pawn_push_tbl_{
  static constexpr size_t num_squares = 64;
  
  piece_type type{piece_type::pawn};
  std::array<square_set, num_squares> data{};

  template<typename T>
  constexpr square_set look_up(const T& sq, const square_set& occ) const {
    static_assert(is_square_v<T>, "can only look up squares");
    square_set occ_ = occ;
    if constexpr(c == color::white){
      occ_.data |= (occ_.data & ~sq.bit_board()) << static_cast<std::uint64_t>(8);
    }else{
      occ_.data |= (occ_.data & ~sq.bit_board()) >> static_cast<std::uint64_t>(8);
    }
    return data[sq.index()] & (~occ_);
  }

  constexpr pawn_push_tbl_(){
    over_all([this](const tbl_square& from){
      if(const tbl_square to = from.add(pawn_delta<c>::step); to.is_valid()){
        data[from.index()].add_(to);
      }
    });
    over_rank(pawn_delta<c>::start_rank_idx, [this](const tbl_square& from){
      const tbl_square to = from.add(pawn_delta<c>::step).add(pawn_delta<c>::step);
      if(to.is_valid()){
        data[from.index()].add_(to);
      }
    });
  }
};

template<typename F, typename D>
constexpr void over_all_slide_masks(const D& deltas, F&& f){
  auto do_ray = [f](const tbl_square from, const delta d){
    for(auto to = from.add(d); to.add(d).is_valid(); to = to.add(d)){
      f(from, to);
    }
  };
  
  over_all([&](const tbl_square from){
    for(const delta d : deltas){ do_ray(from, d); }
  });
}

struct slider_mask_tbl{
  static constexpr size_t num_squares = 64;
  piece_type type;
  std::array<square_set, num_squares> data{};

  template<typename T>
  constexpr const square_set& look_up(const T& sq) const {
    static_assert(is_square_v<T>, "can only look up squares");
    return data[sq.index()];
  }

  template<typename D>
  constexpr slider_mask_tbl(const piece_type pt, const D& deltas) : type{pt} {
    over_all_slide_masks(deltas, [this](const tbl_square& from, const tbl_square& to){
      data[from.index()].add_(to);
    });
  }
};


template<size_t max_num_blockers>
struct slider_attack_tbl{
  static constexpr size_t minor = 64;
  static constexpr size_t major = static_cast<size_t>(1) << max_num_blockers;
  static constexpr std::uint64_t one = static_cast<std::uint64_t>(1);

  piece_type type;
  slider_mask_tbl mask_tbl;
  std::array<square_set, minor*major> data{};

  template<typename T>
  const square_set& look_up(const T& sq, const square_set& blockers) const {
    static_assert(is_square_v<T>, "can only look up squares");
    const square_set mask = mask_tbl.look_up(sq);
    return data[sq.index() * major + pext(blockers.data, mask.data)];
  }
  
  template<typename D>
  constexpr square_set compute_rays(const tbl_square& from, const square_set& blocker, const D& deltas) const {
    square_set result{};
    for(delta d : deltas){
      for(auto to = from.add(d); to.is_valid(); to = to.add(d)){
        result.add_(to);
        if(blocker.occ(to.index())){ break; }
      }
    }
    return result;
  }
  
  template<typename D>
  constexpr slider_attack_tbl(const piece_type pt, const D& deltas) : type{pt}, mask_tbl(pt, deltas) {
    over_all([&, this](const tbl_square from){
      const square_set mask = mask_tbl.look_up(from);
      const std::uint64_t max_blocker = one << pop_count(mask.data);
      for(std::uint64_t blocker_data(0); blocker_data < max_blocker; ++blocker_data){
        const square_set blocker(pdep_compile_time(blocker_data, mask.data));
        data[major * from.index() + blocker_data] = compute_rays(from, blocker, deltas);
      }
    });
  }
};

template<color c>
inline constexpr castle_info_<c> castle_info = castle_info_<c>{};

template<color c>
inline constexpr pawn_push_tbl_<c> pawn_push_tbl = pawn_push_tbl_<c>{};

template<color c>
inline constexpr stepper_attack_tbl pawn_attack_tbl = stepper_attack_tbl{piece_type::pawn, pawn_delta<c>::attack};

template<color c>
inline constexpr passer_tbl_<c> passer_tbl = passer_tbl_<c>{};

inline constexpr stepper_attack_tbl knight_attack_tbl{piece_type::knight, knight_deltas()};
inline constexpr stepper_attack_tbl king_attack_tbl{piece_type::king, king_deltas()};
inline constexpr slider_attack_tbl<9> bishop_attack_tbl{piece_type::bishop, bishop_deltas()};

inline constexpr slider_attack_tbl<12> rook_attack_tbl{piece_type::rook, rook_deltas()};

}
