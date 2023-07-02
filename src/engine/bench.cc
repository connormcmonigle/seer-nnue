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

#include <engine/bench.h>

namespace engine {

std::ostream& operator<<(std::ostream& os, const bench_info& info) noexcept {
  return os << info.total_nodes << " nodes " << info.nodes_per_second << " nps";
}

bench_info get_bench_info(const nnue::weights& weights) noexcept {
  using worker_type = search::search_worker;
  std::shared_ptr<search::search_constants> constants = std::make_shared<search::search_constants>(1);
  std::shared_ptr<search::transposition_table> tt = std::make_shared<search::transposition_table>(bench_config::tt_mb_size);

  std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(&weights, tt, constants, [&](const auto& w) {
    if (w.depth() >= bench_config::bench_depth) { worker->stop(); }
  });

  simple_timer<std::chrono::milliseconds> timer{};
  std::size_t total_nodes{};

  for (const auto& fen : bench_config::fens) {
    worker->go(chess::board_history{}, chess::board::parse_fen(std::string(fen)), bench_config::init_depth);
    worker->iterative_deepening_loop();
    total_nodes += worker->nodes();
  }

  const std::size_t nodes_per_second = total_nodes * std::chrono::milliseconds(std::chrono::seconds(1)).count() / timer.elapsed().count();

  return bench_info{total_nodes, nodes_per_second};
}

std::size_t perft(const chess::board& bd, const search::depth_type& depth) noexcept {
  if (depth == 0) { return bd.generate_moves<>().size(); }
  std::size_t result{0};
  for (const auto& mv : bd.generate_moves<>()) { result += perft(bd.forward(mv), depth - 1); }
  return result;
}

bench_info get_perft_info(const chess::board& bd, const search::depth_type& depth) noexcept {
  simple_timer<std::chrono::nanoseconds> timer{};
  const std::size_t nodes = perft(bd, depth - 1);
  const std::size_t nodes_per_second = nodes * std::chrono::nanoseconds(std::chrono::seconds(1)).count() / timer.elapsed().count();
  return bench_info{nodes, nodes_per_second};
}

}  // namespace engine