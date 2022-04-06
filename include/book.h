/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

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

#include <move.h>
#include <zobrist_util.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace chess {

struct book {
  static constexpr char delimiter = '|';

  std::unordered_map<zobrist::half_hash_type, move::data_type> positions{};

  size_t size() const { return positions.size(); }

  std::optional<move> find(const zobrist::hash_type& key) const {
    const auto it = positions.find(zobrist::upper_half(key));
    if (it != positions.end()) { return std::optional(move(std::get<1>(*it))); }
    return std::nullopt;
  }

  void load(const std::string& path) {
    positions.clear();
    std::fstream reader(path);

    std::string line{};
    while (std::getline(reader, line)) {
      std::stringstream ss(line);
      std::string fen{};
      std::getline(ss, fen, delimiter);
      std::string mv_name{};
      std::getline(ss, mv_name, delimiter);

      const board bd = board::parse_fen(fen);
      const move_list list = bd.generate_moves<>();
      const auto it = std::find_if(list.begin(), list.end(), [=](const move& mv) { return mv.name(bd.turn()) == mv_name; });

      if (it != list.end()) { positions.insert(std::make_pair(zobrist::upper_half(bd.hash()), it->data)); }
    }
  }
};

}  // namespace chess