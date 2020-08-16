#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <cassert>
#include <array>
#include <string>
#include <limits>
#include <vector>
#include <tuple>

#include <zobrist_util.h>
#include <position_history.h>
#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>
#include <manifest.h>
#include <latent.h>



namespace chess{

namespace feature_idx{

constexpr size_t major = 64 * 12;
constexpr size_t minor = 64;

constexpr size_t us_pawn_offset = 0;
constexpr size_t us_knight_offset = us_pawn_offset + minor;
constexpr size_t us_bishop_offset = us_knight_offset + minor;
constexpr size_t us_rook_offset = us_bishop_offset + minor;
constexpr size_t us_queen_offset = us_rook_offset + minor;
constexpr size_t us_king_offset = us_queen_offset + minor;

constexpr size_t them_pawn_offset = us_king_offset + minor;
constexpr size_t them_knight_offset = them_pawn_offset + minor;
constexpr size_t them_bishop_offset = them_knight_offset + minor;
constexpr size_t them_rook_offset = them_bishop_offset + minor;
constexpr size_t them_queen_offset = them_rook_offset + minor;
constexpr size_t them_king_offset = them_queen_offset + minor;

constexpr size_t us_offset(piece_type pt){
  switch(pt){
    case piece_type::pawn: return us_pawn_offset;
    case piece_type::knight: return us_knight_offset;
    case piece_type::bishop: return us_bishop_offset;
    case piece_type::rook: return us_rook_offset;
    case piece_type::queen: return us_queen_offset;
    case piece_type::king: return us_king_offset;
    default: return us_pawn_offset;
  }
}

constexpr size_t them_offset(piece_type pt){
  switch(pt){
    case piece_type::pawn: return them_pawn_offset;
    case piece_type::knight: return them_knight_offset;
    case piece_type::bishop: return them_bishop_offset;
    case piece_type::rook: return them_rook_offset;
    case piece_type::queen: return them_queen_offset;
    case piece_type::king: return them_king_offset;
    default: return them_pawn_offset;
  }
}

}

template<typename T>
constexpr T material_value(const piece_type& pt){
  constexpr std::array<T, 6> values = {100, 300, 300, 450, 900, std::numeric_limits<T>::max()};
  return values[static_cast<size_t>(pt)];
}

struct board{
  sided_manifest man_{};
  sided_latent lat_{};

  bool turn() const {
    return lat_.move_count % 2 == 0;
  }

  zobrist::hash_type hash() const {
    return man_.hash() ^ lat_.hash();
  }

  template<color c>
  std::tuple<piece_type, square> least_valuable_attacker(const square& tgt, const square_set& ignore) const {
    const auto p_mask = pawn_attack_tbl<them_<c>::value>.look_up(tgt);
    const auto p_attackers = p_mask & man_.us<c>().pawn() & ~ignore;
    if(p_attackers.any()){ return std::tuple(piece_type::pawn, *p_attackers.begin()); }

    const auto n_mask = knight_attack_tbl.look_up(tgt);
    const auto n_attackers = n_mask & man_.us<c>().knight() & ~ignore;
    if(n_attackers.any()){ return std::tuple(piece_type::knight, *n_attackers.begin()); }
    
    const square_set occ = (man_.white.all() | man_.black.all()) & ~ignore;
    
    const auto b_mask = bishop_attack_tbl.look_up(tgt, occ);
    const auto b_attackers = b_mask & man_.us<c>().bishop() & ~ignore;
    if(b_attackers.any()){ return std::tuple(piece_type::bishop, *b_attackers.begin()); }
    
    const auto r_mask = rook_attack_tbl.look_up(tgt, occ);
    const auto r_attackers = r_mask & man_.us<c>().rook() & ~ignore;
    if(r_attackers.any()){ return std::tuple(piece_type::rook, *r_attackers.begin()); }
    
    const auto q_mask = b_mask | r_mask;
    const auto q_attackers = q_mask & man_.us<c>().queen() & ~ignore;
    if(q_attackers.any()){ return std::tuple(piece_type::queen, *q_attackers.begin()); }
    
    const auto k_mask = king_attack_tbl.look_up(tgt);
    const auto k_attackers = k_mask & man_.us<c>().king() & ~ignore;
    if(k_attackers.any()){ return std::tuple(piece_type::king, *k_attackers.begin()); }
    
    return std::tuple(piece_type::pawn, tgt);
  }

  template<color c>
  std::tuple<square_set, square_set> checkers(const square_set& occ) const {
    const auto b_check_mask = bishop_attack_tbl.look_up(man_.us<c>().king().item(), occ);
    const auto r_check_mask = rook_attack_tbl.look_up(man_.us<c>().king().item(), occ);
    const auto n_check_mask = knight_attack_tbl.look_up(man_.us<c>().king().item());
    const auto p_check_mask = pawn_attack_tbl<c>.look_up(man_.us<c>().king().item());
    const auto q_check_mask = b_check_mask | r_check_mask;

    const auto b_checkers = (b_check_mask & (man_.them<c>().bishop() | man_.them<c>().queen()));
    const auto r_checkers = (r_check_mask & (man_.them<c>().rook() | man_.them<c>().queen()));
    
    square_set checker_rays_{};
    for(const auto sq : b_checkers){
      checker_rays_ |= bishop_attack_tbl.look_up(sq, occ) & b_check_mask;
    }
    for(const auto sq : r_checkers){
      checker_rays_ |= rook_attack_tbl.look_up(sq, occ) & r_check_mask;
    }
    
    const auto checkers_ =
      (b_check_mask & man_.them<c>().bishop() & occ)|
      (r_check_mask & man_.them<c>().rook() & occ)|
      (n_check_mask & man_.them<c>().knight() & occ)|
      (p_check_mask & man_.them<c>().pawn() & occ)|
      (q_check_mask & man_.them<c>().queen() & occ);
    return std::tuple(checkers_, checker_rays_);
  }

  template<color c>
  square_set king_danger() const {
    const square_set occ = (man_.white.all() | man_.black.all()) & ~man_.us<c>().king();
    square_set k_danger{};
    for(const auto sq : man_.them<c>().pawn()){
      k_danger |= pawn_attack_tbl<them_<c>::value>.look_up(sq);
    }
    for(const auto sq : man_.them<c>().knight()){
      k_danger |= knight_attack_tbl.look_up(sq);
    }
    for(const auto sq : man_.them<c>().king()){
      k_danger |= king_attack_tbl.look_up(sq);
    }
    for(const auto sq : man_.them<c>().rook()){
      k_danger |= rook_attack_tbl.look_up(sq, occ);
    }
    for(const auto sq : man_.them<c>().bishop()){
      k_danger |= bishop_attack_tbl.look_up(sq, occ);
    }
    for(const auto sq : man_.them<c>().queen()){
      k_danger |= rook_attack_tbl.look_up(sq, occ);
      k_danger |= bishop_attack_tbl.look_up(sq, occ);
    }
    return k_danger;
  }

  template<color c>
  square_set pinned() const {
    const square_set occ = man_.white.all() | man_.black.all();
    const auto k_x_diag = bishop_attack_tbl.look_up(man_.us<c>().king().item(), square_set{});
    const auto k_x_hori = rook_attack_tbl.look_up(man_.us<c>().king().item(), square_set{});
    const auto b_check_mask = bishop_attack_tbl.look_up(man_.us<c>().king().item(), occ);
    const auto r_check_mask = rook_attack_tbl.look_up(man_.us<c>().king().item(), occ);
    square_set pinned_set{};
    for(const auto sq : (k_x_hori & (man_.them<c>().queen() | man_.them<c>().rook()))){
      pinned_set |= r_check_mask & rook_attack_tbl.look_up(sq, occ) & man_.us<c>().all();
    }
    for(const auto sq : (k_x_diag & (man_.them<c>().queen() | man_.them<c>().bishop()))){
      pinned_set |= b_check_mask & bishop_attack_tbl.look_up(sq, occ) & man_.us<c>().all();
    }
    return pinned_set;
  }

  template<color c>
  move_list& append_en_passant(move_list& mv_ls) const {
    if(lat_.them<c>().ep_mask().any()){
      const square_set occ = man_.white.all() | man_.black.all();
      const square ep_square = lat_.them<c>().ep_mask().item();
      const square_set enemy_pawn_mask = pawn_push_tbl<them_<c>::value>.look_up(ep_square, square_set{});
      const square_set from_mask = pawn_attack_tbl<them_<c>::value>.look_up(ep_square) & man_.us<c>().pawn();
      for(const auto from : from_mask){
        const square_set occ_ = (occ & ~square_set{from.bit_board()} & ~enemy_pawn_mask) | lat_.them<c>().ep_mask();
        if(!std::get<0>(checkers<c>(occ_)).any()){
          mv_ls.add_(from, ep_square, piece_type::pawn, false, piece_type::pawn, true, enemy_pawn_mask.item());
        }
      }
    }
    return mv_ls;
  }

  template<color c>
  move_list generate_moves_() const {
    move_list result{};
    const square_set occ = man_.white.all() | man_.black.all();
    const auto [checkers_, checker_rays_] = checkers<c>(occ);
    const square_set king_danger_ = king_danger<c>();
    const size_t num_checkers = checkers_.count();
    const auto k_x_diag = bishop_attack_tbl.look_up(man_.us<c>().king().item(), square_set{});
    const auto k_x_hori = rook_attack_tbl.look_up(man_.us<c>().king().item(), square_set{});
    if(num_checkers == 0){
      const square_set pinned_ = pinned<c>();
      for(const auto from : (man_.us<c>().pawn() & ~pinned_)){
        const auto to_quiet = pawn_push_tbl<c>.look_up(from, occ);
        const auto to_loud = pawn_attack_tbl<c>.look_up(from) & man_.them<c>().all();
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::pawn); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
      }
      for(const auto from : (man_.us<c>().knight() & ~pinned_)){
        const auto to_mask = knight_attack_tbl.look_up(from);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::knight); }
        for(const auto to : (to_mask & man_.them<c>().all())){
          result.add_(from, to, piece_type::knight, true, man_.them<c>().occ(to));
        }
      }
      for(const auto from : (man_.us<c>().rook() & ~pinned_)){
        const auto to_mask = rook_attack_tbl.look_up(from, occ);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::rook); }
        for(const auto to : (to_mask & man_.them<c>().all())){
          result.add_(from, to, piece_type::rook, true, man_.them<c>().occ(to));
        }
      }
      for(const auto from : (man_.us<c>().bishop() & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::bishop); }
        for(const auto to : (to_mask & man_.them<c>().all())){
          result.add_(from, to, piece_type::bishop, true, man_.them<c>().occ(to));
        }
      }
      for(const auto from : (man_.us<c>().queen() & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ) | rook_attack_tbl.look_up(from, occ);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::queen); }
        for(const auto to : (to_mask & man_.them<c>().all())){
          result.add_(from, to, piece_type::queen, true, man_.them<c>().occ(to));
        }
      }
      if(lat_.us<c>().oo() && !(castle_info<c>.oo_mask & (king_danger_ | occ)).any()){
        result.add_(castle_info<c>.start_king, castle_info<c>.oo_rook, piece_type::king, true, piece_type::rook);
      }
      if(lat_.us<c>().ooo() && !(castle_info<c>.ooo_danger_mask & king_danger_).any() && !(castle_info<c>.ooo_occ_mask & occ).any()){
        result.add_(castle_info<c>.start_king, castle_info<c>.ooo_rook, piece_type::king, true, piece_type::rook);
      }
      if(pinned_.any()){
        for(const auto from : (man_.us<c>().pawn() & pinned_ & k_x_diag)){
          const auto to_mask = pawn_attack_tbl<c>.look_up(from) & k_x_diag;
          for(const auto to : (to_mask & man_.them<c>().all())){
            result.add_(from, to, piece_type::pawn, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().pawn() & pinned_ & k_x_hori)){
          const auto to_mask = pawn_push_tbl<c>.look_up(from, occ) & k_x_hori;
          for(const auto to : to_mask){
            result.add_(from, to, piece_type::pawn);
          }
        }
        for(const auto from : (man_.us<c>().bishop() & pinned_ & k_x_diag)){
          const auto to_mask = bishop_attack_tbl.look_up(from, occ) & k_x_diag;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::bishop); }
          for(const auto to : (to_mask & man_.them<c>().all())){
            result.add_(from, to, piece_type::bishop, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().rook() & pinned_ & k_x_hori)){
          const auto to_mask = rook_attack_tbl.look_up(from, occ) & k_x_hori;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::rook); }
          for(const auto to : (to_mask & man_.them<c>().all())){
            result.add_(from, to, piece_type::rook, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().queen() & pinned_ & k_x_diag)){
          const auto to_mask = bishop_attack_tbl.look_up(from, occ) & k_x_diag;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::queen); }
          for(const auto to : (to_mask & man_.them<c>().all())){
            result.add_(from, to, piece_type::queen, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().queen() & pinned_ & k_x_hori)){
          const auto to_mask = rook_attack_tbl.look_up(from, occ) & k_x_hori;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::queen); }
          for(const auto to : (to_mask & man_.them<c>().all())){
            result.add_(from, to, piece_type::queen, true, man_.them<c>().occ(to));
          }
        }
      }
    }else if(num_checkers == 1){
      const square_set pinned_ = pinned<c>();
      const square_set push_mask = checker_rays_;
      const square_set capture_mask = checkers_;
      const piece_type checker_type = man_.them<c>().occ(capture_mask.item());
      for(const auto from : (man_.us<c>().pawn() & ~pinned_)){
        const auto to_quiet = push_mask & pawn_push_tbl<c>.look_up(from, occ);
        const auto to_loud = capture_mask & pawn_attack_tbl<c>.look_up(from);
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::pawn); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::pawn, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().knight() & ~pinned_)){
        const auto to_mask = knight_attack_tbl.look_up(from);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::knight); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::knight, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().rook() & ~pinned_)){
        const auto to_mask = rook_attack_tbl.look_up(from, occ);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::rook); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::rook, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().bishop() & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::bishop); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::bishop, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().queen() & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ) | rook_attack_tbl.look_up(from, occ);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::queen); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::queen, true, checker_type); }
      }
    }
    const square_set to_mask = ~king_danger_ & king_attack_tbl.look_up(man_.us<c>().king().item());
    for(const square to : (to_mask & ~occ)){
      result.add_(man_.us<c>().king().item(), to, piece_type::king);
    }
    for(const square to : (to_mask & man_.them<c>().all())){
      result.add_(man_.us<c>().king().item(), to, piece_type::king, true, man_.them<c>().occ(to));
    }
    return append_en_passant<c>(result);
  }

  move_list generate_moves() const {
    return turn() ? generate_moves_<color::white>() : generate_moves_<color::black>();
  }

  template<color c>
  bool is_check_() const {
    return std::get<0>(checkers<c>(man_.white.all() | man_.black.all())).any();
  }

  bool is_check() const {
    return turn() ? is_check_<color::white>() : is_check_<color::black>();
  }
  
  template<color c, typename T>
  T see_(const move& mv) const {
    const square tgt_sq = mv.to();
    size_t last_idx{0};
    std::array<T, 32> material_deltas{};
    auto used_mask = square_set{};
    auto on_sq = mv.piece();
    used_mask.add_(mv.from());

    for(;;){
      {
        const auto [p, sq] = least_valuable_attacker<them_<c>::value>(tgt_sq, used_mask);
        if(sq == tgt_sq){ break; }
        
        material_deltas[last_idx++] = material_value<T>(on_sq);
        used_mask.add_(sq);
        on_sq = p;
      }

      {
        const auto [p, sq] = least_valuable_attacker<c>(tgt_sq, used_mask);
        if(sq == tgt_sq){ break; }
        
        material_deltas[last_idx++] = material_value<T>(on_sq);
        used_mask.add_(sq);
        on_sq = p;
      }
    }
    
    T delta_sum{};
    for(auto iter = material_deltas.rend() - last_idx; iter != material_deltas.rend(); ++iter){
      delta_sum = std::max(T{}, *iter - delta_sum);
    }

    const T base = (mv.is_capture() &&
      !mv.is_castle_ooo<c>() &&
      !mv.is_castle_oo<c>()) ?
        material_value<T>(mv.captured()) :
        T{};
    return base - delta_sum;
  }

  template<typename T>
  T see(const move& mv) const {
    return turn() ? see_<chess::color::white, T>(mv) : see_<chess::color::black, T>(mv);
  }

  template<color c>
  board forward_(const move& mv) const {
    auto cpy = *this;
    if(mv.is_castle_ooo<c>()){
      cpy.lat_.us<c>().set_ooo(false).set_oo(false);
      cpy.man_.us<c>().remove_piece(piece_type::king, castle_info<c>.start_king);
      cpy.man_.us<c>().remove_piece(piece_type::rook, castle_info<c>.ooo_rook);
      cpy.man_.us<c>().add_piece(piece_type::king, castle_info<c>.after_ooo_king);
      cpy.man_.us<c>().add_piece(piece_type::rook, castle_info<c>.after_ooo_rook);
    }else if(mv.is_castle_oo<c>()){
      cpy.lat_.us<c>().set_ooo(false).set_oo(false);
      cpy.man_.us<c>().remove_piece(piece_type::king, castle_info<c>.start_king);
      cpy.man_.us<c>().remove_piece(piece_type::rook, castle_info<c>.oo_rook);
      cpy.man_.us<c>().add_piece(piece_type::king, castle_info<c>.after_oo_king);
      cpy.man_.us<c>().add_piece(piece_type::rook, castle_info<c>.after_oo_rook);
    }else{
      cpy.man_.us<c>().remove_piece(mv.piece(), mv.from());
      if(mv.is_promotion<c>()){
        cpy.man_.us<c>().add_piece(piece_type::queen, mv.to());
      }else{
        cpy.man_.us<c>().add_piece(mv.piece(), mv.to());
      }
      if(mv.is_capture()){
        cpy.man_.them<c>().remove_piece(mv.captured(), mv.to());
      }else if(mv.is_enpassant()){
        cpy.man_.them<c>().remove_piece(piece_type::pawn, mv.enpassant_sq());
      }else if(mv.is_pawn_double<c>()){
        cpy.lat_.us<c>().set_ep_mask(pawn_push_tbl<them_<c>::value>.look_up(mv.to(), square_set{}).item());
      }
      if(mv.from() == castle_info<c>.start_king){
        cpy.lat_.us<c>().set_ooo(false).set_oo(false);
      }else if(mv.from() == castle_info<c>.oo_rook){
        cpy.lat_.us<c>().set_oo(false);
      }else if(mv.from() == castle_info<c>.ooo_rook){
        cpy.lat_.us<c>().set_ooo(false);
      }
      if(mv.to() == castle_info<them_<c>::value>.oo_rook){
        cpy.lat_.them<c>().set_oo(false);
      }else if(mv.to() == castle_info<them_<c>::value>.ooo_rook){
        cpy.lat_.them<c>().set_ooo(false);
      }
    }
    cpy.lat_.them<c>().clear_ep_mask();
    ++cpy.lat_.move_count;
    return cpy;
  }

  board forward(const move& mv) const {
    return turn() ? forward_<color::white>(mv) : forward_<color::black>(mv);
  }

  template<color c, typename U>
  void show_half_kp_indices(U& updatable) const {
    using namespace feature_idx;
    const size_t king_idx = man_.us<c>().king().item().index();
    //us
    for(const auto sq : man_.us<c>().pawn()){ updatable.template us<c>().insert(major*king_idx + us_pawn_offset + sq.index()); }
    for(const auto sq : man_.us<c>().knight()){ updatable.template us<c>().insert(major*king_idx + us_knight_offset + sq.index()); }
    for(const auto sq : man_.us<c>().bishop()){ updatable.template us<c>().insert(major*king_idx + us_bishop_offset + sq.index()); }
    for(const auto sq : man_.us<c>().rook()){ updatable.template us<c>().insert(major*king_idx + us_rook_offset + sq.index()); }
    for(const auto sq : man_.us<c>().queen()){ updatable.template us<c>().insert(major*king_idx + us_queen_offset + sq.index()); }
    for(const auto sq : man_.us<c>().king()){ updatable.template us<c>().insert(major*king_idx + us_king_offset + sq.index()); }
    //them
    for(const auto sq : man_.them<c>().pawn()){ updatable.template us<c>().insert(major*king_idx + them_pawn_offset + sq.index()); }
    for(const auto sq : man_.them<c>().knight()){ updatable.template us<c>().insert(major*king_idx + them_knight_offset + sq.index()); }
    for(const auto sq : man_.them<c>().bishop()){ updatable.template us<c>().insert(major*king_idx + them_bishop_offset + sq.index()); }
    for(const auto sq : man_.them<c>().rook()){ updatable.template us<c>().insert(major*king_idx + them_rook_offset + sq.index()); }
    for(const auto sq : man_.them<c>().queen()){ updatable.template us<c>().insert(major*king_idx + them_queen_offset + sq.index()); }
    for(const auto sq : man_.them<c>().king()){ updatable.template us<c>().insert(major*king_idx + them_king_offset + sq.index()); }
  }

  template<typename U>
  void show_init(U& u) const {
    u.white.clear();
    u.black.clear();
    show_half_kp_indices<color::white>(u);
    show_half_kp_indices<color::black>(u);
  }

  template<color c, typename U>
  void show_delta(const move& mv, U& updatable) const {
    using namespace feature_idx;
    const size_t their_king_idx = man_.them<c>().king().item().index();
    const size_t our_king_idx = man_.us<c>().king().item().index();
    if(mv.piece() == piece_type::king){
      forward_<c>(mv).show_init(updatable);
    }else{
      updatable.template us<c>().erase(major * our_king_idx + mv.from().index() + us_offset(mv.piece()));
      updatable.template them<c>().erase(major * their_king_idx + mv.from().index() + them_offset(mv.piece()));
      if(mv.is_promotion<c>()){
        updatable.template us<c>().insert(major * our_king_idx + mv.to().index() + us_queen_offset);
        updatable.template them<c>().insert(major * their_king_idx + mv.to().index() + them_queen_offset);
      }else{
        updatable.template us<c>().insert(major * our_king_idx + mv.to().index() + us_offset(mv.piece()));
        updatable.template them<c>().insert(major * their_king_idx + mv.to().index() + them_offset(mv.piece()));
      }
      if(mv.is_enpassant()){
        updatable.template them<c>().erase(major * their_king_idx + mv.enpassant_sq().index() + us_pawn_offset);
        updatable.template us<c>().erase(major * our_king_idx + mv.enpassant_sq().index() + them_pawn_offset);
      }
      if(mv.is_capture()){
        updatable.template them<c>().erase(major * their_king_idx + mv.to().index() + us_offset(mv.captured()));
        updatable.template us<c>().erase(major * our_king_idx + mv.to().index() + them_offset(mv.captured()));
      }
    }
  }

  template<typename U>
  U half_kp_updated(const move& mv, const U& updatable) const {
    U u = updatable;
    if(turn()){ show_delta<color::white, U>(mv, u); } else { show_delta<color::black, U>(mv, u); }
    return u;
  }

  std::tuple<position_history, board> after_uci_moves(const std::string& moves) const {
    position_history history{};
    auto bd = *this;
    std::istringstream move_stream(moves);
    std::string move_name;
    while(move_stream >> move_name){
      const move_list list = bd.generate_moves();
      const auto it = std::find_if(list.cbegin(), list.cend(), [=](const move& mv){
        return mv.name(bd.turn()) == move_name;
      });
      assert((it != list.cend()));
      history.push_(bd.hash());
      bd = bd.forward(*it);
    }
    return std::tuple(history, bd);
  }

  std::string fen() const {
    std::string fen{};
    constexpr size_t num_ranks = 8;
    for(size_t i{0}; i < num_ranks; ++i){
      size_t j{0};
      over_rank(i, [&, this](const tbl_square& at_r){
        const tbl_square at = at_r.rotated();
        if(man_.white.all().occ(at.index())){
          const char letter = piece_letter(color::white, man_.white.occ(at));
          if(j != 0){ fen.append(std::to_string(j)); }
          fen.push_back(letter);
          j = 0;
        }else if(man_.black.all().occ(at.index())){
          const char letter = piece_letter(color::black, man_.black.occ(at));
          if(j != 0){
            fen.append(std::to_string(j));
          }
          fen.push_back(letter);
          j = 0;
        }else{
          ++j;
        }
      });
      if(j != 0){ fen.append(std::to_string(j)); }
      if(i != (num_ranks - 1)){ fen.push_back('/'); }
    }
    fen.push_back(' ');
    fen.push_back(turn() ? 'w' : 'b');
    fen.push_back(' ');
    std::string castle_rights{};
    if(lat_.white.oo()){ castle_rights.push_back('K'); }
    if(lat_.white.ooo()){ castle_rights.push_back('Q'); }
    if(lat_.black.oo()){ castle_rights.push_back('k'); }
    if(lat_.black.ooo()){ castle_rights.push_back('q'); }
    fen.append(castle_rights.empty() ? "-" : castle_rights);
    fen.push_back(' ');
    fen.append(lat_.them(turn()).ep_mask().any() ? lat_.them(turn()).ep_mask().item().name() : "-");
    fen.push_back(' ');
    fen.append(std::to_string(lat_.half_clock));
    fen.push_back(' ');
    fen.append(std::to_string(1 + (lat_.move_count / 2)));
    return fen;
  }

  static board start_pos(){
    return parse_fen("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  }

  static board parse_fen(const std::string& fen){
    auto fen_pos = board();
    std::stringstream ss(fen);
    
    std::string body; ss >> body;
    std::string side; ss >> side;
    std::string castle; ss >> castle;
    std::string ep_sq; ss >> ep_sq;
    std::string half_clock; ss >> half_clock;
    std::string ply; ss >> ply;
    {
      std::stringstream body_s(body);
      std::string rank;
      for(int rank_idx{0}; std::getline(body_s, rank, '/'); ++rank_idx){
        int file_idx{0};
        for(const char c : rank){
          if(std::isdigit(c)){
            file_idx += static_cast<int>(c - '0');
          }else{
            const color side = color_from(c);
            const piece_type type = type_from(c);
            const tbl_square sq = tbl_square{file_idx, rank_idx}.rotated();
            fen_pos.man_.us(side).add_piece(type, sq);
            ++file_idx;
          }
        }
      }
    }
    fen_pos.lat_.white.set_oo(castle.find('K') != std::string::npos);
    fen_pos.lat_.white.set_ooo(castle.find('Q') != std::string::npos);
    fen_pos.lat_.black.set_oo(castle.find('k') != std::string::npos);
    fen_pos.lat_.black.set_ooo(castle.find('q') != std::string::npos);
    fen_pos.lat_.half_clock = std::stol(half_clock);
    if(ep_sq != "-"){
      fen_pos.lat_.them(side == "w").set_ep_mask(tbl_square::from_name(ep_sq));
    }
    fen_pos.lat_.move_count = static_cast<size_t>(2 * (std::stoi(ply) - 1) + static_cast<int>(side != "w"));
    return fen_pos;
  }

};

std::ostream& operator<<(std::ostream& ostr, const board& bd){
  ostr << std::boolalpha;
  ostr << "board(hash=" << bd.hash();
  ostr << ", half_clock=" << bd.lat_.half_clock;
  ostr << ", move_count=" << bd.lat_.move_count;
  ostr << ", white.oo_=" << bd.lat_.white.oo();
  ostr << ", white.ooo_=" << bd.lat_.white.ooo();
  ostr << ", black.oo_=" << bd.lat_.black.oo();
  ostr << ", black.ooo_=" << bd.lat_.black.ooo();
  ostr << ",\nwhite.ep_mask=" << bd.lat_.white.ep_mask();
  ostr << ",\nblack.ep_mask=" << bd.lat_.black.ep_mask();
  ostr << "white.occ_table={";
  over_all([&ostr, bd](const tbl_square& sq){
    ostr << piece_name(bd.man_.white.occ(sq)) << ", ";
  });
  ostr << "},\nblack.occ_table={";
  over_all([&ostr, bd](const tbl_square& sq){
    ostr << piece_name(bd.man_.black.occ(sq)) << ", ";
  });
  ostr << "}\n";
  over_types([&ostr, bd](const piece_type& pt){
    ostr << "white." << piece_name(pt) << "=" << bd.man_.white.get_plane(pt) << ",\n";
  });
  ostr << "white.all=" << bd.man_.white.all() << ",\n";
  over_types([&ostr, bd](const piece_type& pt){
    ostr << "black." << piece_name(pt) << "=" << bd.man_.black.get_plane(pt) << ",\n";
  });
  ostr << "black.all=" << bd.man_.black.all() << ")";
  return ostr << std::noboolalpha;
}

}
