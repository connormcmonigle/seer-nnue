#pragma once

#include <iostream>
#include <sstream>
#include <vector>
#include <cstdint>
#include <array>
#include <string>
#include <limits>
#include <tuple>


#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>


namespace chess{

struct manifest{
  static constexpr size_t num_squares = 64;
  std::array<piece_type, num_squares> occ_table{};
  square_set pawn{};
  square_set knight{};
  square_set bishop{};
  square_set rook{};
  square_set queen{};
  square_set king{};
  square_set all{};

  const piece_type& occ(const square& sq) const {
    return occ_table[sq.index()];
  }

  piece_type& occ(const square& sq){
    return occ_table[sq.index()];
  }

  square_set& get_plane(const piece_type pt){
    return get_member(pt, *this);
  }

  const square_set& get_plane(const piece_type pt) const {
    return get_member(pt, *this);
  }

  square_set& get_all(){ return all; }
  const square_set& get_all() const { return all; }
};

struct latent{
  bool oo_{true};
  bool ooo_{true};
  square_set ep_mask{};
};

struct sided_manifest : sided<sided_manifest, manifest> {
  manifest white{};
  manifest black{};
  sided_manifest(){}
};

struct sided_latent : sided<sided_latent, latent> {
  size_t half_clock{0};
  size_t move_count{0};
  latent white{};
  latent black{};
  sided_latent(){}
};

struct board{
  sided_manifest man_{};
  sided_latent lat_{};

  bool turn() const {
    return lat_.move_count % 2 == 0;
  }

  template<color c>
  std::tuple<square_set, square_set> checkers(const square_set& occ) const {
    const auto b_check_mask = bishop_attack_tbl.look_up(man_.us<c>().king.item(), occ);
    const auto r_check_mask = rook_attack_tbl.look_up(man_.us<c>().king.item(), occ);
    const auto n_check_mask = knight_attack_tbl.look_up(man_.us<c>().king.item());
    const auto p_check_mask = pawn_attack_tbl<c>.look_up(man_.us<c>().king.item());
    const auto q_check_mask = b_check_mask | r_check_mask;

    const auto b_checkers = (b_check_mask & (man_.them<c>().bishop | man_.them<c>().queen));
    const auto r_checkers = (r_check_mask & (man_.them<c>().rook | man_.them<c>().queen));
    
    square_set checker_rays_{};
    for(const auto sq : b_checkers){
      checker_rays_ |= bishop_attack_tbl.look_up(sq, occ) & b_check_mask;
    }
    for(const auto sq : r_checkers){
      checker_rays_ |= rook_attack_tbl.look_up(sq, occ) & r_check_mask;
    }
    
    const auto checkers_ =
      (b_check_mask & man_.them<c>().bishop & occ)|
      (r_check_mask & man_.them<c>().rook & occ)|
      (n_check_mask & man_.them<c>().knight & occ)|
      (p_check_mask & man_.them<c>().pawn & occ)|
      (q_check_mask & man_.them<c>().queen & occ);
    return std::tuple(checkers_, checker_rays_);
  }

  template<color c>
  square_set king_danger() const {
    const square_set occ = (man_.white.get_all() | man_.black.get_all()) & ~man_.us<c>().king;
    square_set k_danger{};
    for(const auto sq : man_.them<c>().pawn){
      k_danger |= pawn_attack_tbl<them_<c>::value>.look_up(sq);
    }
    for(const auto sq : man_.them<c>().knight){
      k_danger |= knight_attack_tbl.look_up(sq);
    }
    for(const auto sq : man_.them<c>().king){
      k_danger |= king_attack_tbl.look_up(sq);
    }
    for(const auto sq : man_.them<c>().rook){
      k_danger |= rook_attack_tbl.look_up(sq, occ);
    }
    for(const auto sq : man_.them<c>().bishop){
      k_danger |= bishop_attack_tbl.look_up(sq, occ);
    }
    for(const auto sq : man_.them<c>().queen){
      k_danger |= rook_attack_tbl.look_up(sq, occ);
      k_danger |= bishop_attack_tbl.look_up(sq, occ);
    }
    return k_danger;
  }

  template<color c>
  square_set pinned() const {
    const square_set occ = man_.white.get_all() | man_.black.get_all();
    const auto k_x_diag = bishop_attack_tbl.look_up(man_.us<c>().king.item(), square_set{});
    const auto k_x_hori = rook_attack_tbl.look_up(man_.us<c>().king.item(), square_set{});
    const auto b_check_mask = bishop_attack_tbl.look_up(man_.us<c>().king.item(), occ);
    const auto r_check_mask = rook_attack_tbl.look_up(man_.us<c>().king.item(), occ);
    square_set pinned_set{};
    for(const auto sq : (k_x_hori & (man_.them<c>().queen | man_.them<c>().rook))){
      pinned_set |= r_check_mask & rook_attack_tbl.look_up(sq, occ) & man_.us<c>().get_all();
    }
    for(const auto sq : (k_x_diag & (man_.them<c>().queen | man_.them<c>().bishop))){
      pinned_set |= b_check_mask & bishop_attack_tbl.look_up(sq, occ) & man_.us<c>().get_all();
    }
    return pinned_set;
  }

  template<color c>
  move_list& append_en_passant(move_list& mv_ls) const {
    //en_passant captures are rarely available -> amadahl's law?
    if(lat_.them<c>().ep_mask.any()){
      const square_set occ = man_.white.get_all() | man_.black.get_all();
      const square ep_square = lat_.them<c>().ep_mask.item();
      const square_set enemy_pawn_mask = pawn_push_tbl<them_<c>::value>.look_up(ep_square, square_set{});
      const square_set from_mask = pawn_attack_tbl<them_<c>::value>.look_up(ep_square) & man_.us<c>().pawn;
      for(const auto from : from_mask){
        const square_set occ_ = (occ & ~square_set{from.bit_board()} & ~enemy_pawn_mask) | lat_.them<c>().ep_mask;
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
    const square_set occ = man_.white.get_all() | man_.black.get_all();
    const auto [checkers_, checker_rays_] = checkers<c>(occ);
    const square_set king_danger_ = king_danger<c>();
    const size_t num_checkers = checkers_.count();
    const auto k_x_diag = bishop_attack_tbl.look_up(man_.us<c>().king.item(), square_set{});
    const auto k_x_hori = rook_attack_tbl.look_up(man_.us<c>().king.item(), square_set{});
    if(num_checkers == 0){
      const square_set pinned_ = pinned<c>();
      for(const auto from : (man_.us<c>().pawn & ~pinned_)){
        const auto to_quiet = pawn_push_tbl<c>.look_up(from, occ);
        const auto to_loud = pawn_attack_tbl<c>.look_up(from) & man_.them<c>().get_all();
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::pawn); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::pawn, true, man_.them<c>().occ(to)); }
      }
      for(const auto from : (man_.us<c>().knight & ~pinned_)){
        const auto to_mask = knight_attack_tbl.look_up(from);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::knight); }
        for(const auto to : (to_mask & man_.them<c>().get_all())){
          result.add_(from, to, piece_type::knight, true, man_.them<c>().occ(to));
        }
      }
      for(const auto from : (man_.us<c>().rook & ~pinned_)){
        const auto to_mask = rook_attack_tbl.look_up(from, occ);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::rook); }
        for(const auto to : (to_mask & man_.them<c>().get_all())){
          result.add_(from, to, piece_type::rook, true, man_.them<c>().occ(to));
        }
      }
      for(const auto from : (man_.us<c>().bishop & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::bishop); }
        for(const auto to : (to_mask & man_.them<c>().get_all())){
          result.add_(from, to, piece_type::bishop, true, man_.them<c>().occ(to));
        }
      }
      for(const auto from : (man_.us<c>().queen & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ) | rook_attack_tbl.look_up(from, occ);
        for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::queen); }
        for(const auto to : (to_mask & man_.them<c>().get_all())){
          result.add_(from, to, piece_type::queen, true, man_.them<c>().occ(to));
        }
      }
      if(lat_.us<c>().oo_ && !(castle_info<c>.oo_mask & (king_danger_ | occ)).any()){
        result.add_(castle_info<c>.start_king, castle_info<c>.oo_rook, piece_type::king, true, piece_type::rook);
      }
      if(lat_.us<c>().ooo_ && !(castle_info<c>.ooo_danger_mask & king_danger_).any() && !(castle_info<c>.ooo_occ_mask & occ).any()){
        result.add_(castle_info<c>.start_king, castle_info<c>.ooo_rook, piece_type::king, true, piece_type::rook);
      }
      if(pinned_.any()){
        for(const auto from : (man_.us<c>().pawn & pinned_ & k_x_diag)){
          const auto to_mask = pawn_attack_tbl<c>.look_up(from) & k_x_diag;
          for(const auto to : (to_mask & man_.them<c>().get_all())){
            result.add_(from, to, piece_type::pawn, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().pawn & pinned_ & k_x_hori)){
          const auto to_mask = pawn_push_tbl<c>.look_up(from, occ) & k_x_hori;
          for(const auto to : to_mask){
            result.add_(from, to, piece_type::pawn);
          }
        }
        for(const auto from : (man_.us<c>().bishop & pinned_ & k_x_diag)){
          const auto to_mask = bishop_attack_tbl.look_up(from, occ) & k_x_diag;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::bishop); }
          for(const auto to : (to_mask & man_.them<c>().get_all())){
            result.add_(from, to, piece_type::bishop, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().rook & pinned_ & k_x_hori)){
          const auto to_mask = rook_attack_tbl.look_up(from, occ) & k_x_hori;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::rook); }
          for(const auto to : (to_mask & man_.them<c>().get_all())){
            result.add_(from, to, piece_type::rook, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().queen & pinned_ & k_x_diag)){
          const auto to_mask = bishop_attack_tbl.look_up(from, occ) & k_x_diag;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::queen); }
          for(const auto to : (to_mask & man_.them<c>().get_all())){
            result.add_(from, to, piece_type::queen, true, man_.them<c>().occ(to));
          }
        }
        for(const auto from : (man_.us<c>().queen & pinned_ & k_x_hori)){
          const auto to_mask = rook_attack_tbl.look_up(from, occ) & k_x_hori;
          for(const auto to : (to_mask & ~occ)){ result.add_(from, to, piece_type::queen); }
          for(const auto to : (to_mask & man_.them<c>().get_all())){
            result.add_(from, to, piece_type::queen, true, man_.them<c>().occ(to));
          }
        }
      }
    }else if(num_checkers == 1){
      const square_set pinned_ = pinned<c>();
      const square_set push_mask = checker_rays_;
      const square_set capture_mask = checkers_;
      const piece_type checker_type = man_.them<c>().occ(capture_mask.item());
      for(const auto from : (man_.us<c>().pawn & ~pinned_)){
        const auto to_quiet = push_mask & pawn_push_tbl<c>.look_up(from, occ);
        const auto to_loud = capture_mask & pawn_attack_tbl<c>.look_up(from);
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::pawn); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::pawn, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().knight & ~pinned_)){
        const auto to_mask = knight_attack_tbl.look_up(from);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::knight); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::knight, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().rook & ~pinned_)){
        const auto to_mask = rook_attack_tbl.look_up(from, occ);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::rook); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::rook, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().bishop & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::bishop); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::bishop, true, checker_type); }
      }
      for(const auto from : (man_.us<c>().queen & ~pinned_)){
        const auto to_mask = bishop_attack_tbl.look_up(from, occ) | rook_attack_tbl.look_up(from, occ);
        const auto to_quiet = push_mask & to_mask;
        const auto to_loud = capture_mask & to_mask;
        for(const auto to : to_quiet){ result.add_(from, to, piece_type::queen); }
        for(const auto to : to_loud){ result.add_(from, to, piece_type::queen, true, checker_type); }
      }
    }
    const square_set to_mask = ~king_danger_ & king_attack_tbl.look_up(man_.us<c>().king.item());
    for(const square to : (to_mask & ~occ)){
      result.add_(man_.us<c>().king.item(), to, piece_type::king);
    }
    for(const square to : (to_mask & man_.them<c>().get_all())){
      result.add_(man_.us<c>().king.item(), to, piece_type::king, true, man_.them<c>().occ(to));
    }
    return append_en_passant<c>(result);
  }

  move_list generate_moves() const {
    return turn() ? generate_moves_<color::white>() : generate_moves_<color::black>();
  }

  template<color c>
  board forward_(const move& mv) const {
    auto cpy = *this;
    if(mv.is_castle_ooo<c>()){
      cpy.lat_.us<c>().ooo_ = cpy.lat_.us<c>().oo_ = false;
      cpy.man_.us<c>().king &= ~castle_info<c>.start_king.bit_board();
      cpy.man_.us<c>().rook &= ~castle_info<c>.ooo_rook.bit_board();
      cpy.man_.us<c>().get_all() &= ~castle_info<c>.start_king.bit_board();
      cpy.man_.us<c>().get_all() &= ~castle_info<c>.ooo_rook.bit_board();
      cpy.man_.us<c>().king |= castle_info<c>.after_ooo_king.bit_board();
      cpy.man_.us<c>().rook |= castle_info<c>.after_ooo_rook.bit_board();
      cpy.man_.us<c>().get_all() |= castle_info<c>.after_ooo_king.bit_board();
      cpy.man_.us<c>().get_all() |= castle_info<c>.after_ooo_rook.bit_board();
      cpy.man_.us<c>().occ(castle_info<c>.after_ooo_king) = piece_type::king;
      cpy.man_.us<c>().occ(castle_info<c>.after_ooo_rook) = piece_type::rook;
    }else if(mv.is_castle_oo<c>()){
      cpy.lat_.us<c>().ooo_ = cpy.lat_.us<c>().oo_ = false;
      cpy.man_.us<c>().king &= ~castle_info<c>.start_king.bit_board();
      cpy.man_.us<c>().rook &= ~castle_info<c>.oo_rook.bit_board();
      cpy.man_.us<c>().get_all() &= ~castle_info<c>.start_king.bit_board();
      cpy.man_.us<c>().get_all() &= ~castle_info<c>.oo_rook.bit_board();
      cpy.man_.us<c>().king |= castle_info<c>.after_oo_king.bit_board();
      cpy.man_.us<c>().rook |= castle_info<c>.after_oo_rook.bit_board();
      cpy.man_.us<c>().get_all() |= castle_info<c>.after_oo_king.bit_board();
      cpy.man_.us<c>().get_all() |= castle_info<c>.after_oo_rook.bit_board();
      cpy.man_.us<c>().occ(castle_info<c>.after_oo_king) = piece_type::king;
      cpy.man_.us<c>().occ(castle_info<c>.after_oo_rook) = piece_type::rook;
    }else{
      cpy.man_.us<c>().get_plane(mv.piece()) &= ~mv.from().bit_board();
      cpy.man_.us<c>().get_all() &= ~mv.from().bit_board();
      cpy.man_.us<c>().get_all() |= mv.to().bit_board();
      if(mv.is_promotion<c>()){
        cpy.man_.us<c>().get_plane(piece_type::queen) |= mv.to().bit_board();
        cpy.man_.us<c>().occ(mv.to()) = piece_type::queen;
      }else{
        cpy.man_.us<c>().get_plane(mv.piece()) |= mv.to().bit_board();
        std::swap(cpy.man_.us<c>().occ(mv.from()), cpy.man_.us<c>().occ(mv.to()));
      }
      if(mv.is_capture()){
        cpy.man_.them<c>().get_plane(mv.captured()) &= ~mv.to().bit_board();
        cpy.man_.them<c>().get_all() &= ~mv.to().bit_board();
      }else if(mv.is_enpassant()){
        cpy.man_.them<c>().pawn &= ~mv.enpassant_sq().bit_board();
        cpy.man_.them<c>().get_all() &= ~mv.enpassant_sq().bit_board();
      }else if(mv.is_pawn_double<c>()){
        cpy.lat_.us<c>().ep_mask = pawn_push_tbl<them_<c>::value>.look_up(mv.to(), square_set{});
      }
      if(mv.from() == castle_info<c>.start_king){
        cpy.lat_.us<c>().ooo_ = cpy.lat_.us<c>().oo_ = false;
      }else if(mv.from() == castle_info<c>.oo_rook){
        cpy.lat_.us<c>().oo_ = false;
      }else if(mv.from() == castle_info<c>.ooo_rook){
        cpy.lat_.us<c>().ooo_ = false;
      }
      if(mv.to() == castle_info<them_<c>::value>.oo_rook){
        cpy.lat_.them<c>().oo_ = false;
      }else if(mv.to() == castle_info<them_<c>::value>.ooo_rook){
        cpy.lat_.them<c>().ooo_ = false;
      }
    }
    cpy.lat_.them<c>().ep_mask = square_set{};
    ++cpy.lat_.move_count;
    return cpy;
  }

  board forward(const move& mv) const {
    return turn() ? forward_<color::white>(mv) : forward_<color::black>(mv);
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
            file_idx += std::atoi(&c);
          }else{
            const color side = color_from(c);
            const piece_type type = type_from(c);
            const tbl_square sq = tbl_square{file_idx, rank_idx}.rotated();
            fen_pos.man_.us(side).get_plane(type).add_(sq);
            fen_pos.man_.us(side).get_all().add_(sq);
            fen_pos.man_.us(side).occ_table[sq.index()] = type;
            ++file_idx;
          }
        }
      }
    }
    fen_pos.lat_.white.oo_ = castle.find('K') != std::string::npos;
    fen_pos.lat_.white.ooo_ = castle.find('Q') != std::string::npos;
    fen_pos.lat_.black.oo_ = castle.find('k') != std::string::npos;
    fen_pos.lat_.black.ooo_ = castle.find('q') != std::string::npos;
    fen_pos.lat_.half_clock = std::stoi(half_clock);
    if(ep_sq != "-"){
      fen_pos.lat_.them(side == "w").ep_mask.add_(tbl_square::from_name(ep_sq));
    }
    fen_pos.lat_.move_count = 2 * (std::stoi(ply) - 1) + static_cast<size_t>(side != "w");
    return fen_pos;
  }

};

std::ostream& operator<<(std::ostream& ostr, const board& bd){
  ostr << std::boolalpha;
  ostr << "board(half_clock=" << bd.lat_.half_clock;
  ostr << ", move_count=" << bd.lat_.move_count;
  ostr << ", white.oo_=" << bd.lat_.white.oo_;
  ostr << ", white.ooo_=" << bd.lat_.white.ooo_;
  ostr << ", black.oo_=" << bd.lat_.black.oo_;
  ostr << ", black.ooo_=" << bd.lat_.black.ooo_;
  ostr << ",\nwhite.ep_mask=" << bd.lat_.white.ep_mask;
  ostr << ",\nblack.ep_mask=" << bd.lat_.black.ep_mask;
  ostr << "white.occ_table={";
  for(size_t i(0); i < bd.man_.white.occ_table.size()-1; ++i){
    ostr << piece_name(bd.man_.white.occ_table[i]) << ", ";
  }
  ostr << piece_name(*bd.man_.white.occ_table.rbegin()) << "},\n";
  ostr << "black.occ_table={";
  for(size_t i(0); i < bd.man_.black.occ_table.size()-1; ++i){
    ostr << piece_name(bd.man_.black.occ_table[i]) << ", ";
  }
  ostr << piece_name(*bd.man_.black.occ_table.rbegin()) << "},\n";
  over_types([&ostr, bd](const piece_type& pt){
    ostr << "white." << piece_name(pt) << "=" << bd.man_.white.get_plane(pt) << ",\n";
  });
  ostr << "white.all=" << bd.man_.white.get_all() << ",\n";
  over_types([&ostr, bd](const piece_type& pt){
    ostr << "black." << piece_name(pt) << "=" << bd.man_.black.get_plane(pt) << ",\n";
  });
  ostr << "black.all=" << bd.man_.black.get_all() << ")";
  return ostr << std::noboolalpha;
}

}
