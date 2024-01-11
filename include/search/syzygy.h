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

#pragma once

#include <chess/board.h>
#include <search/search_constants.h>
#include <tbprobe.h>

#include <string>

namespace search::syzygy {

enum class wdl_type { loss, draw, win };

struct tb_wdl_result {
  bool success{};
  wdl_type wdl{wdl_type::draw};

  [[nodiscard]] static constexpr tb_wdl_result failure() noexcept { return tb_wdl_result{false}; }
  [[nodiscard]] static constexpr tb_wdl_result from_value(const unsigned int& value) noexcept {
    if (value == TB_WIN) { return tb_wdl_result{true, wdl_type::win}; }
    if (value == TB_DRAW) { return tb_wdl_result{true, wdl_type::draw}; }
    if (value == TB_LOSS) { return tb_wdl_result{true, wdl_type::loss}; }
    return failure();
  }
};

struct tb_dtz_result {
  bool success{};
  search::score_type score{search::draw_score};
  chess::move move{chess::move::null()};

  [[nodiscard]] static constexpr tb_dtz_result failure() noexcept { return tb_dtz_result{false}; }
  [[nodiscard]] static tb_dtz_result from_value(const chess::board& bd, const unsigned int& value) noexcept;
};

[[nodiscard]] tb_wdl_result probe_wdl(const chess::board& bd) noexcept;
[[nodiscard]] tb_dtz_result probe_dtz(const chess::board& bd) noexcept;

void init(const std::string& path) noexcept;

}  // namespace search::syzygy
