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

#include <uci.h>
#include <mcts.h>

#include <chrono>
#include <iostream>
#include <string>

int main() {
  auto constants = std::make_shared<search::search_constants>();
  auto tt = std::make_shared<search::transposition_table>(128);
  auto weights = std::make_shared<nnue::weights>();
  nnue::embedded_weight_streamer embedded(embed::weights_file_data);
  weights->load(embedded);

  auto walker = std::make_unique<mcts::tree_walker>(weights.get(), tt, constants);

  std::string fen{};
  std::getline(std::cin, fen);
  const auto state = chess::board::parse_fen(fen);
  const auto history = chess::position_history{};

  walker->worker_.go(history, state, mcts::parameters::ab_search_start_depth);
  walker->worker_.iterative_deepening_loop();

  mcts::tree_node tree(nullptr, mcts::index_type{}, walker->worker_.policy<mcts::probability_type>());

  const chess::move_list list = state.generate_moves<chess::generation_mode::all>();
  for (size_t i(0); i < 8192; ++i) {
    walker->walk(history, state, &tree); 
    std::cout << list[tree.best_index()].name(state.turn()) << std::endl;
    std::cout << tree.q_value() << std::endl;
  }
}
