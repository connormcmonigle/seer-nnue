#pragma once

#include <array>
#include <limits>
#include <vector>
#include <tuple>

#include <zobrist_util.h>
#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>

namespace chess{

struct manifest_zobrist_src{
  static constexpr size_t num_squares = 64;
  using plane_t = std::array<zobrist::hash_type, num_squares>;
  plane_t pawn_{};
  plane_t knight_{};
  plane_t bishop_{};
  plane_t rook_{};
  plane_t queen_{};
  plane_t king_{};

  std::array<zobrist::hash_type, num_squares>& get_plane(const piece_type& pt){
    return get_member(pt, *this);
  }

  const std::array<zobrist::hash_type, num_squares>& get_plane(const piece_type& pt) const {
    return get_member(pt, *this);
  }

  template<typename S>
  zobrist::hash_type get(const piece_type& pt, const S& at) const {
    static_assert(is_square_v<S>, "at must be of square type");
    return get_plane(pt)[at.index()];
  }

  manifest_zobrist_src(){
    over_types([this](const piece_type pt){
      plane_t& pt_plane = get_plane(pt);
      std::transform(pt_plane.begin(), pt_plane.end(), pt_plane.begin(), [](auto...){ return zobrist::random_bit_string(); });
    });
  }
};

struct manifest{
  static constexpr size_t num_squares = 64;

  const manifest_zobrist_src* zobrist_src_;
  zobrist::hash_type hash_{0};
  std::array<piece_type, num_squares> occ_table{};
  square_set pawn_{};
  square_set knight_{};
  square_set bishop_{};
  square_set rook_{};
  square_set queen_{};
  square_set king_{};
  square_set all_{};

  zobrist::hash_type hash() const {
    return hash_;
  }

  template<typename S>
  piece_type& occ(const S& sq){
    static_assert(is_square_v<S>, "at must be of square type");
    return occ_table[sq.index()];
  }

  square_set& get_plane(const piece_type pt){
    return get_member(pt, *this);
  }

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
    return get_member(pt, *this);
  }

  template<typename S>
  manifest& add_piece(const piece_type& pt, const S& at){
    static_assert(is_square_v<S>, "at must be of square type");
    hash_ ^= zobrist_src_ -> get(pt, at);
    all_ |= at.bit_board();
    get_plane(pt) |= at.bit_board();
    occ(at) = pt;
    return *this;
  }

  template<typename S>
  manifest& remove_piece(const piece_type& pt, const S& at){
    static_assert(is_square_v<S>, "at must be of square type");
    hash_ ^= zobrist_src_ -> get(pt, at);
    all_ &= ~at.bit_board();
    get_plane(pt) &= ~at.bit_board();
    return *this;
  }

  manifest(const manifest_zobrist_src* src) : zobrist_src_{src} {}
};

struct sided_manifest : sided<sided_manifest, manifest> {
  static inline const manifest_zobrist_src w_manifest_src{};
  static inline const manifest_zobrist_src b_manifest_src{};
  
  manifest white;
  manifest black;

  zobrist::hash_type hash() const {
    return white.hash() ^ black.hash();
  }

  sided_manifest() : white(&w_manifest_src), black(&b_manifest_src) {}
};

}
