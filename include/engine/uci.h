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
#include <engine/time_manager.h>
#include <nnue/weights.h>
#include <search/search_worker.h>
#include <search/search_worker_orchestrator.h>

#include <atomic>
#include <chrono>
#include <iostream>
#include <string>
#include <string_view>

namespace engine {

struct uci {
  static constexpr std::string_view embedded_weight_path = "EMBEDDED";
  static constexpr std::string_view unused_weight_path = "UNUSED";

  static constexpr std::size_t default_thread_count = 1;
  static constexpr std::size_t default_hash_size = 16;
  static constexpr bool default_ponder = false;

  chess::board_history history{};
  chess::board position = chess::board::start_pos();

  nnue::quantized_weights weights_{};
  search::worker_orchestrator orchestrator_;

  std::atomic_bool ponder_{false};
  std::atomic_bool should_quit_{false};

  time_manager manager_;
  simple_timer<std::chrono::milliseconds> timer_;

  std::mutex mutex_{};
  std::ostream& os = std::cout;

  [[nodiscard]] auto options() noexcept;
  [[nodiscard]] bool should_quit() const noexcept;

  void quit() noexcept;

  void new_game() noexcept;
  void set_position(const chess::board& bd, const std::string& uci_moves = "") noexcept;

  void weights_info_string() noexcept;
  void info_string(const search::search_worker& worker) noexcept;

  template <typename T, typename... Ts>
  void init_time_manager(Ts&&... args) noexcept;
  void go() noexcept;

  void ponder_hit() noexcept;
  void stop() noexcept;

  void ready() noexcept;
  void id_info() noexcept;

  void bench() noexcept;
  void eval() noexcept;
  void probe() noexcept;
  void perft(const search::depth_type& depth) noexcept;
  void export_weights(const std::string& export_path) noexcept;

  void read(const std::string& line) noexcept;

  uci() noexcept;
};

}  // namespace engine
