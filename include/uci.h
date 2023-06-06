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

#include <bench.h>
#include <board.h>
#include <embedded_weights.h>
#include <move.h>
#include <option_parser.h>
#include <search_constants.h>
#include <search_stack.h>
#include <search_worker.h>
#include <search_worker_orchestrator.h>
#include <syzygy.h>
#include <time_manager.h>
#include <version.h>

#include <atomic>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>

namespace engine {

struct uci {
  static constexpr size_t default_thread_count = 1;
  static constexpr size_t default_hash_size = 16;
  static constexpr std::string_view default_weight_path = "EMBEDDED";
  static constexpr std::string_view default_syzygy_path = "<empty>";
  static constexpr bool default_ponder = false;

  chess::position_history history{};
  chess::board position = chess::board::start_pos();

  nnue::weights weights_{};
  search::worker_orchestrator orchestrator_;

  std::atomic_bool ponder_{false};
  std::atomic_bool should_quit_{false};

  time_manager manager_;
  simple_timer<std::chrono::milliseconds> timer_;

  std::mutex os_mutex_{};
  std::ostream& os = std::cout;

  bool should_quit() const { return should_quit_.load(); }

  void weights_info_string() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << "info string loaded weights with signature 0x" << std::hex << weights_.signature() << std::dec << std::endl;
  }

  auto options() {
    auto weight_path = option_callback(string_option("Weights", std::string(default_weight_path)), [this](const std::string& path) {
      if (path == std::string(default_weight_path)) {
        nnue::embedded_weight_streamer embedded(embed::weights_file_data);
        weights_.load(embedded);
      } else {
        weights_.load(path);
      }
      weights_info_string();
    });

    auto hash_size = option_callback(spin_option("Hash", default_hash_size, spin_range{1, 262144}), [this](const int size) {
      const auto new_size = static_cast<size_t>(size);
      orchestrator_.tt_->resize(new_size);
    });

    auto thread_count = option_callback(spin_option("Threads", default_thread_count, spin_range{1, 512}), [this](const int count) {
      const auto new_count = static_cast<size_t>(count);
      orchestrator_.resize(new_count);
    });

    auto ponder = option_callback(check_option("Ponder", default_ponder), [this](const bool& value) { ponder_.store(value); });

    auto syzygy_path = option_callback(string_option("SyzygyPath", std::string(default_syzygy_path)), [](const std::string& path) { syzygy::init(path); });

    return uci_options(weight_path, hash_size, thread_count, ponder, syzygy_path);
  }

  void uci_new_game() {
    history.clear();
    position = chess::board::start_pos();
    orchestrator_.reset();
  }

  void set_position(const std::string& line) {
    const std::regex startpos_with_moves("position startpos moves((?: [a-h][1-8][a-h][1-8]+(?:q|r|b|n)?)+)");
    const std::regex fen_with_moves("position fen (.*) moves((?: [a-h][1-8][a-h][1-8]+(?:q|r|b|n)?)+)");
    const std::regex startpos("position startpos");
    const std::regex fen("position fen (.*)");
    std::smatch matches{};

    if (std::regex_search(line, matches, startpos_with_moves)) {
      auto [h_, p_] = chess::board::start_pos().after_uci_moves(matches.str(1));
      history = h_;
      position = p_;
    } else if (std::regex_search(line, matches, fen_with_moves)) {
      position = chess::board::parse_fen(matches.str(1));
      auto [h_, p_] = position.after_uci_moves(matches.str(2));
      history = h_;
      position = p_;
    } else if (std::regex_search(line, matches, startpos)) {
      history.clear();
      position = chess::board::start_pos();
    } else if (std::regex_search(line, matches, fen)) {
      history.clear();
      position = chess::board::parse_fen(matches.str(1));
    }
  }

  template <typename T>
  void info_string(const T& worker) {
    constexpr search::score_type raw_multiplier = 288;
    constexpr search::score_type raw_divisor = 1024;

    std::lock_guard<std::mutex> os_lk(os_mutex_);

    const search::score_type raw_score = worker.score();
    const search::score_type score = raw_score * raw_multiplier / raw_divisor;

    const search::depth_type depth = worker.depth();
    const size_t elapsed_ms = timer_.elapsed().count();
    const size_t nodes = orchestrator_.nodes();
    const size_t tb_hits = orchestrator_.tb_hits();
    const size_t nps = std::chrono::milliseconds(std::chrono::seconds(1)).count() * nodes / (1 + elapsed_ms);

    const bool should_report = orchestrator_.is_searching() && depth < search::max_depth;
    if (should_report) {
      os << "info depth " << depth << " seldepth " << worker.internal.stack.sel_depth() << " score cp " << score << " nodes " << nodes << " nps "
         << nps << " time " << elapsed_ms << " tbhits " << tb_hits << " pv " << worker.internal.stack.pv_string() << std::endl;
    }
  }

  void go(const std::string& line) {
    manager_.init(position.turn(), line);
    timer_.lap();
    orchestrator_.go(history, position);
  }

  void ponder_hit() { manager_.ponder_hit(); }

  void stop() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);

    orchestrator_.stop();
    const chess::move best_move = orchestrator_.primary_worker().best_move();
    const chess::move ponder_move = orchestrator_.primary_worker().ponder_move();

     const std::string ponder_move_string = [&] {
      if (!position.forward(best_move).is_legal<chess::generation_mode::all>(ponder_move)) { return std::string{}; }
      return std::string(" ponder ") + ponder_move.name(position.forward(best_move).turn());
    }();

    os << "bestmove " << best_move.name(position.turn()) << ponder_move_string << std::endl;
  }

  void ready() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << "readyok" << std::endl;
  }

  void id_info() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << "id name " << version::engine_name << " " << version::major << '.' << version::minor << '.' << version::patch << std::endl;
    os << "id author " << version::author_name << std::endl;
    os << options();
    if constexpr (search::search_constants::tuning) { os << orchestrator_.constants_->options(); }

    os << "uciok" << std::endl;
  }

  void bench() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << get_bench_info(weights_) << std::endl;
  }

  void eval() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);

    auto scratchpad = std::make_unique<nnue::eval::scratchpad_type>();
    auto evaluator = nnue::eval(&weights_, scratchpad.get(), 0, 0);
    position.feature_full_reset(evaluator);

    os << "phase: " << position.phase<nnue::weights::parameter_type>() << std::endl;
    os << "score(phase): " << evaluator.evaluate(position.turn(), position.phase<nnue::weights::parameter_type>()) << std::endl;
  }

  void probe() {
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    if (position.is_rule50_draw()) {
      std::cout << "rule 50 draw" << std::endl;
    } else if (const syzygy::tb_wdl_result result = syzygy::probe_wdl(position); result.success) {
      std::cout << "success: " << [&] {
        switch (result.wdl) {
          case syzygy::wdl_type::loss: return "loss";
          case syzygy::wdl_type::draw: return "draw";
          case syzygy::wdl_type::win: return "win";
          default: return "unknown";
        }
      }() << std::endl;
    } else {
      std::cout << "fail" << std::endl;
    }
  }

  void perft(const std::string& line) {
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    const std::regex perft_with_depth("perft ([0-9]+)");
    if (std::smatch matches{}; std::regex_search(line, matches, perft_with_depth)) {
      const search::depth_type depth = std::stoi(matches.str(1));
      std::cout << get_perft_info(position, depth) << std::endl;
    }
  }

  void quit() { should_quit_.store(true); }

  void read(const std::string& line) {
    const std::regex position_rgx("position(.*)");
    const std::regex go_rgx("go(.*)");
    const std::regex perft_rgx("perft(.*)");

    const bool is_searching = orchestrator_.is_searching();
    if (!is_searching && line == "uci") {
      id_info();
    } else if (line == "isready") {
      ready();
    } else if (!is_searching && line == "ucinewgame") {
      uci_new_game();
    } else if (is_searching && line == "stop") {
      stop();
    } else if (is_searching && line == "ponderhit") {
      ponder_hit();
    } else if (line == "_internal_board") {
      os << position << std::endl;
    } else if (!is_searching && line == "bench") {
      bench();
    } else if (!is_searching && line == "eval") {
      eval();
    } else if (!is_searching && line == "probe") {
      probe();
    } else if (!is_searching && std::regex_match(line, perft_rgx)) {
      perft(line);
    } else if (!is_searching && std::regex_match(line, go_rgx)) {
      go(line);
    } else if (!is_searching && std::regex_match(line, position_rgx)) {
      set_position(line);
    } else if (line == "quit") {
      quit();
    } else if (!is_searching) {
      options().update(line);
      if constexpr (search::search_constants::tuning) { orchestrator_.constants_->options().update(line); }
    }
  }

  uci()
      : orchestrator_(
            &weights_,
            default_hash_size,
            [this](const auto& worker) {
              info_string(worker);
              if (manager_.should_stop_on_iter(iter_info{worker.depth(), worker.best_move_percent()})) { stop(); }
            },
            [this](const auto& worker) {
              if (manager_.should_stop_on_update(update_info{worker.nodes()})) { stop(); }
            }) {
    nnue::embedded_weight_streamer embedded(embed::weights_file_data);
    weights_.load(embedded);
    orchestrator_.resize(default_thread_count);
  }
};

}  // namespace engine
