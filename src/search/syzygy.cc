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

#include <search/syzygy.h>

namespace search {
namespace syzygy {

tb_dtz_result tb_dtz_result::from_value(const chess::board& bd, const unsigned int& value) noexcept {
  auto is_same_promo = [](const chess::move& mv, const int& promo) {
    constexpr int num_pieces = 6;
    return ((!mv.is_promotion() && promo == 0) || (mv.is_promotion() && (num_pieces - promo - 1) == static_cast<int>(mv.promotion())));
  };

  if (value == TB_RESULT_FAILED || value == TB_RESULT_CHECKMATE || value == TB_RESULT_STALEMATE) { return failure(); }
  const int wdl = TB_GET_WDL(value);

  const chess::move dtz_move = [&] {
    const chess::move_list list = bd.generate_moves<>();
    const int promo = TB_GET_PROMOTES(value);
    const int from = TB_GET_FROM(value);
    const int to = TB_GET_TO(value);
    const auto it = std::find_if(list.begin(), list.end(), [&](const chess::move& mv) {
      return mv.from().index() == from && mv.to().index() == to && is_same_promo(mv, promo);
    });
    if (it != list.end()) { return *it; }
    return chess::move::null();
  }();

  if (dtz_move == chess::move::null()) { return failure(); }

  if (wdl == TB_WIN) { return tb_dtz_result{true, search::tb_win_score, dtz_move}; }
  if (wdl == TB_LOSS) { return tb_dtz_result{true, search::tb_loss_score, dtz_move}; }
  return tb_dtz_result{true, search::draw_score, dtz_move};
}

tb_wdl_result probe_wdl(const chess::board& bd) noexcept {
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

tb_dtz_result probe_dtz(const chess::board& bd) noexcept {
  if (bd.num_pieces() > TB_LARGEST) { return tb_dtz_result::failure(); }
  if (bd.lat_.white.oo() || bd.lat_.white.ooo() || bd.lat_.black.oo() || bd.lat_.black.ooo()) { return tb_dtz_result::failure(); }

  const unsigned int rule_50 = bd.lat_.half_clock;
  constexpr unsigned int castling_rights = 0;
  const unsigned int ep = bd.lat_.them(bd.turn()).ep_mask().any() ? bd.lat_.them(bd.turn()).ep_mask().item().index() : 0;
  const bool turn = bd.turn();

  const unsigned value = tb_probe_root(
      bd.man_.white.all().data, bd.man_.black.all().data, (bd.man_.white.king() | bd.man_.black.king()).data,
      (bd.man_.white.queen() | bd.man_.black.queen()).data, (bd.man_.white.rook() | bd.man_.black.rook()).data,
      (bd.man_.white.bishop() | bd.man_.black.bishop()).data, (bd.man_.white.knight() | bd.man_.black.knight()).data,
      (bd.man_.white.pawn() | bd.man_.black.pawn()).data, rule_50, castling_rights, ep, turn, nullptr);

  return tb_dtz_result::from_value(bd, value);
}

void init(const std::string& path) noexcept { tb_init(path.c_str()); }

}  // namespace syzygy
}  // namespace search