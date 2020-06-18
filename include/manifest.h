#pragma once

#include <array>
#include <limits>
#include <vector>
#include <tuple>

#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>

namespace chess{

struct manifest{
  static constexpr size_t num_squares = 64;
 private:
  std::array<piece_type, num_squares> occ_table{};
  square_set pawn_{};
  square_set knight_{};
  square_set bishop_{};
  square_set rook_{};
  square_set queen_{};
  square_set king_{};
  square_set all_{};

  template<typename S>
  piece_type& occ(const S& sq){
    static_assert(is_square_v<S>, "at must be of square type");
    return occ_table[sq.index()];
  }

  square_set& get_plane(const piece_type pt){
    switch(pt){
      case piece_type::pawn: return pawn_;
      case piece_type::knight: return knight_;
      case piece_type::bishop: return bishop_;
      case piece_type::rook: return rook_;
      case piece_type::queen: return queen_;
      case piece_type::king: return king_;
      default: return king_;
    }
  }

 public:
  template<typename S>
  const piece_type& occ(const S& at) const {
    static_assert(is_square_v<S>, "at must be of square type");
    return occ_table[at.index()];
  }

  const square_set& all() const { return all_; }
  const square_set& pawn() const { return pawn_; }
  const square_set& knight() const { return knight_; }
  const square_set& bishop() const { return bishop_; }
  const square_set& rook() const { return rook_; }
  const square_set& queen() const { return queen_; }
  const square_set& king() const { return king_; }

  const square_set& get_plane(const piece_type pt) const {
    switch(pt){
      case piece_type::pawn: return pawn_;
      case piece_type::knight: return knight_;
      case piece_type::bishop: return bishop_;
      case piece_type::rook: return rook_;
      case piece_type::queen: return queen_;
      case piece_type::king: return king_;
      default: return king_;
    }
  }

  template<typename S>
  manifest& add_piece(const piece_type& pt, const S& at){
    static_assert(is_square_v<S>, "at must be of square type");
    all_ |= at.bit_board();
    get_plane(pt) |= at.bit_board();
    occ(at) = pt;
    return *this;
  }

  template<typename S>
  manifest& remove_piece(const piece_type& pt, const S& at){
    static_assert(is_square_v<S>, "at must be of square type");
    all_ &= ~at.bit_board();
    get_plane(pt) &= ~at.bit_board();
    return *this;
  }
};

}