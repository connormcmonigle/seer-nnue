#pragma once

#include <board.h>
#include <search_constants.h>
#include <tbprobe.h>
#include <transposition_table.h>

#include <string>

namespace syzygy {
struct tb_wdl_result {
  bool success;
  chess::bound_type bound{chess::bound_type::exact};
  search::score_type score{search::draw_score};

  static constexpr tb_wdl_result failure() { return tb_wdl_result{false}; }
  static constexpr tb_wdl_result from_value(const unsigned int& value) {
    if (value == TB_WIN) { return tb_wdl_result{true, chess::bound_type::lower, search::tb_win_score}; }
    if (value == TB_LOSS) { return tb_wdl_result{true, chess::bound_type::upper, search::tb_loss_score}; }
    if (value == TB_DRAW) { return tb_wdl_result{true, chess::bound_type::exact, search::draw_score}; }
    return failure();
  }
};

tb_wdl_result probe_wdl(const chess::board& bd) {
  if (bd.num_pieces() > TB_LARGEST || bd.lat_.half_clock != 0) { return tb_wdl_result::failure(); }
  if (bd.lat_.white.oo() || bd.lat_.white.ooo() || bd.lat_.black.oo() || bd.lat_.black.ooo()) { return tb_wdl_result::failure(); }

  constexpr unsigned int rule_50 = 0;
  constexpr unsigned int castling_rights = 0;
  const unsigned int ep = bd.lat_.them(bd.turn()).ep_mask().any() ? bd.lat_.them(bd.turn()).ep_mask().item().index() : 0;
  const bool turn = bd.turn();

  const unsigned value = tb_probe_wdl(
      bd.man_.white.all().data, bd.man_.black.all().data, (bd.man_.white.king() | bd.man_.black.king()).data,
      (bd.man_.white.queen() | bd.man_.black.queen()).data, (bd.man_.white.rook() | bd.man_.black.rook()).data,
      (bd.man_.white.bishop() | bd.man_.black.bishop()).data, (bd.man_.white.knight() | bd.man_.black.knight()).data,
      (bd.man_.white.pawn() | bd.man_.black.pawn()).data, rule_50, castling_rights, ep, turn);

  return tb_wdl_result::from_value(value);
}

void init(const std::string& path) { tb_init(path.c_str()); }

}  // namespace syzygy