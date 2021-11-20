#pragma once

#include <chess_types.h>
#include <square.h>

namespace feature {

namespace half_ka {
constexpr size_t numel = 768 * 64;
constexpr size_t max_active_half_features = 32;

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

constexpr size_t us_offset(const chess::piece_type& pt) {
  switch (pt) {
    case chess::piece_type::pawn: return us_pawn_offset;
    case chess::piece_type::knight: return us_knight_offset;
    case chess::piece_type::bishop: return us_bishop_offset;
    case chess::piece_type::rook: return us_rook_offset;
    case chess::piece_type::queen: return us_queen_offset;
    case chess::piece_type::king: return us_king_offset;
    default: return us_pawn_offset;
  }
}

constexpr size_t them_offset(const chess::piece_type& pt) {
  switch (pt) {
    case chess::piece_type::pawn: return them_pawn_offset;
    case chess::piece_type::knight: return them_knight_offset;
    case chess::piece_type::bishop: return them_bishop_offset;
    case chess::piece_type::rook: return them_rook_offset;
    case chess::piece_type::queen: return them_queen_offset;
    case chess::piece_type::king: return them_king_offset;
    default: return them_pawn_offset;
  }
}

template <chess::color us, typename T>
struct mapper {
  T* sided_set;
  chess::square their_king;
  chess::square our_king;

  template <chess::color point_of_view>
  constexpr chess::square king() const {
    if constexpr (us == point_of_view) {
      return our_king;
    } else {
      return their_king;
    }
  }

  template <chess::color point_of_view, chess::color piece_color>
  constexpr size_t index_(const chess::piece_type& type, const chess::square& sq) const {
    if constexpr (point_of_view == piece_color) {
      return major * king<point_of_view>().index() + us_offset(type) + sq.index();
    } else {
      return major * king<point_of_view>().index() + them_offset(type) + sq.index();
    }
  }

  template <chess::color point_of_view>
  void clear() const {
    sided_set->template us<point_of_view>().clear();
  }

  template <chess::color point_of_view, chess::color piece_color>
  void insert(const chess::piece_type& type, const chess::square& sq) const {
    sided_set->template us<point_of_view>().insert(index_<point_of_view, piece_color>(type, sq));
  }

  template <chess::color point_of_view, chess::color piece_color>
  void erase(const chess::piece_type& type, const chess::square& sq) const {
    sided_set->template us<point_of_view>().erase(index_<point_of_view, piece_color>(type, sq));
  }
};

}  // namespace half_ka

}  // namespace feature