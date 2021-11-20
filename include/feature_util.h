#pragma once

#include <chess_types.h>
#include <square.h>

namespace feature {

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

template <chess::color us, chess::color p>
constexpr size_t index(const chess::square& ks, const chess::piece_type& pt, const chess::square& sq) {
  if constexpr (us == p) {
    return major * ks.index() + us_offset(pt) + sq.index();
  } else {
    return major * ks.index() + them_offset(pt) + sq.index();
  }
}

}  // namespace feature