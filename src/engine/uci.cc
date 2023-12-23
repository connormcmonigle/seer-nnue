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

#include <chess/move.h>
#include <engine/bench.h>
#include <engine/option_parser.h>
#include <engine/processor/types.h>
#include <engine/uci.h>
#include <engine/version.h>
#include <nnue/embedded_weights.h>
#include <nnue/weights_exporter.h>
#include <nnue/weights_streamer.h>
#include <search/search_constants.h>
#include <search/syzygy.h>

#include <sstream>

namespace engine {

auto uci::options() noexcept {
  auto quantized_weight_path = option_callback(string_option("QuantizedWeights", std::string(embedded_weight_path)), [this](const std::string& path) {
    if (path == std::string(embedded_weight_path)) {
      nnue::embedded_weight_streamer embedded(nnue::embed::weights_file_data);
      weights_.load(embedded);
    } else {
      weights_.load(path);
    }

    weights_info_string();
  });

  auto weight_path = option_callback(string_option("Weights", std::string(unused_weight_path)), [this](const std::string& path) {
    if (path == std::string(unused_weight_path)) { return; }

    nnue::weights raw_weights{};
    raw_weights.load(path);

    weights_ = raw_weights.to<nnue::quantized_weights>();
    weights_info_string();
  });

  auto hash_size = option_callback(spin_option("Hash", default_hash_size, spin_range{1, 262144}), [this](const int size) {
    const auto new_size = static_cast<std::size_t>(size);
    orchestrator_.tt_->resize(new_size);
  });

  auto thread_count = option_callback(spin_option("Threads", default_thread_count, spin_range{1, 512}), [this](const int count) {
    const auto new_count = static_cast<std::size_t>(count);
    orchestrator_.resize(new_count);
  });

  auto ponder = option_callback(check_option("Ponder", default_ponder), [this](const bool& value) { ponder_.store(value); });
  auto syzygy_path = option_callback(string_option("SyzygyPath", string_option::empty), [](const std::string& path) { search::syzygy::init(path); });

  return uci_options(quantized_weight_path, weight_path, hash_size, thread_count, ponder, syzygy_path);
}

bool uci::should_quit() const noexcept { return should_quit_.load(); }
void uci::quit() noexcept { should_quit_.store(true); }

void uci::new_game() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  history.clear();
  position = chess::board::start_pos();
  orchestrator_.reset();
}

void uci::set_position(const chess::board& bd, const std::string& uci_moves) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  auto [history_value, position_value] = bd.after_uci_moves(uci_moves);
  history = history_value;
  position = position_value;
}

void uci::weights_info_string() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  os << "info string loaded weights with signature 0x" << std::hex << weights_.signature() << std::dec << std::endl;
}

void uci::info_string(const search::search_worker& worker) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);

  constexpr search::score_type raw_multiplier = 288;
  constexpr search::score_type raw_divisor = 1024;

  const search::score_type raw_score = worker.score();
  const search::score_type score = raw_score * raw_multiplier / raw_divisor;

  const search::depth_type depth = worker.depth();
  const std::size_t elapsed_ms = timer_.elapsed().count();
  const std::size_t nodes = orchestrator_.nodes();
  const std::size_t tb_hits = orchestrator_.tb_hits();
  const std::size_t nps = std::chrono::milliseconds(std::chrono::seconds(1)).count() * nodes / (1 + elapsed_ms);

  const bool should_report = orchestrator_.is_searching() && depth < search::max_depth;
  if (should_report) {
    os << "info depth " << depth << " seldepth " << worker.internal.stack.selective_depth() << " score cp " << score << " nodes " << nodes << " nps "
       << nps << " time " << elapsed_ms << " tbhits " << tb_hits << " pv " << worker.internal.stack.pv_string() << std::endl;
  }
}

template <typename T, typename... Ts>
void uci::init_time_manager(Ts&&... args) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  manager_.init(position.turn(), T{std::forward<Ts>(args)...});
}

void uci::go() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  timer_.lap();
  orchestrator_.go(history, position);
}

void uci::ponder_hit() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!orchestrator_.is_searching()) { return; }

  manager_.ponder_hit();
}

void uci::stop() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (!orchestrator_.is_searching()) { return; }

  orchestrator_.stop();
  const chess::move best_move = orchestrator_.primary_worker().best_move();
  const chess::move ponder_move = orchestrator_.primary_worker().ponder_move();

  const std::string ponder_move_string = [&] {
    if (!position.forward(best_move).is_legal<chess::generation_mode::all>(ponder_move)) { return std::string{}; }
    return std::string(" ponder ") + ponder_move.name(position.forward(best_move).turn());
  }();

  os << "bestmove " << best_move.name(position.turn()) << ponder_move_string << std::endl;
}

void uci::ready() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  os << "readyok" << std::endl;
}

void uci::id_info() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  os << "id name " << version::engine_name << " " << version::major << '.' << version::minor << '.' << version::patch << std::endl;
  os << "id author " << version::author_name << std::endl;
  os << options();
  os << orchestrator_.constants()->options();
  os << "uciok" << std::endl;
}

void uci::tune_config() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }
  os << orchestrator_.constants_->options().ob_spsa_config() << std::endl;
}

void uci::bench() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  os << get_bench_info(weights_) << std::endl;
}

void uci::export_weights(const std::string& export_path) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  nnue::weights_exporter exporter(export_path);
  weights_.write(exporter);
}

void uci::eval() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  auto scratchpad = std::make_unique<nnue::eval::scratchpad_type>();
  auto evaluator = nnue::eval(&weights_, scratchpad.get(), 0, 0);
  position.feature_full_reset(evaluator);

  os << "phase: " << position.phase<nnue::weights::parameter_type>() << std::endl;
  os << "score(phase): " << evaluator.evaluate(position.turn(), position.phase<nnue::weights::parameter_type>()) << std::endl;
}

void uci::probe() noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }

  if (position.is_rule50_draw()) {
    os << "rule 50 draw" << std::endl;
  } else if (const search::syzygy::tb_wdl_result result = search::syzygy::probe_wdl(position); result.success) {
    os << "success: " << [&] {
      switch (result.wdl) {
        case search::syzygy::wdl_type::loss: return "loss";
        case search::syzygy::wdl_type::draw: return "draw";
        case search::syzygy::wdl_type::win: return "win";
        default: return "unknown";
      }
    }() << std::endl;
  } else {
    os << "fail" << std::endl;
  }
}

void uci::perft(const search::depth_type& depth) noexcept {
  std::lock_guard<std::mutex> lock(mutex_);
  if (orchestrator_.is_searching()) { return; }
  os << get_perft_info(position, depth) << std::endl;
}

void uci::read(const std::string& line) noexcept {
  using namespace processor::def;

  // clang-format off
  
  const auto processor = parallel(
    sequential(consume("uci"), invoke([&] { id_info(); })),
    sequential(consume("isready"), invoke([&] { ready(); })),

    options().processor(),
    orchestrator_.constants()->options().processor(),

    sequential(consume("ucinewgame"), invoke([&] { new_game(); })),
    sequential(consume("position"), parallel(
      sequential(consume("startpos"), parallel(
        sequential(invoke([&] { set_position(chess::board::start_pos()); })),
        sequential(consume("moves"), emit_all, invoke([&] (const std::string& moves) { set_position(chess::board::start_pos(), moves); }))
      )),

      sequential(consume("fen"), emit_n<chess::board::num_fen_tokens>, parallel(
        sequential(invoke([&] (const std::string& fen) {
          const auto board = chess::board::parse_fen(fen);
          set_position(board);
        })),
        
        sequential(consume("moves"), emit_all, invoke([&] (const std::string& fen, const std::string& moves) {
          const auto board = chess::board::parse_fen(fen);
          set_position(board, moves); 
        }))
      ))
    )),

    sequential(consume("go"), parallel(
      sequential(consume("infinite"), invoke([&] { init_time_manager<go::infinite>(); })),      
      sequential(consume("nodes"), emit<std::size_t>, invoke([&] (const std::size_t& nodes) { init_time_manager<go::nodes>(nodes); })),
      sequential(consume("depth"), emit<search::depth_type>, invoke([&] (const search::depth_type& depth) { init_time_manager<go::depth>(depth); })),

      sequential(condition("ponder"), parallel(
        sequential(key<int>("movetime"), invoke([&](const auto& ... args) { init_time_manager<go::move_time>(args...); })),
        
        sequential(key<int>("wtime"), key<int>("btime"), parallel(
          sequential(invoke([&](const auto& ... args) { init_time_manager<go::sudden_death>(args...); })),
          sequential(key<int>("movestogo"), invoke([&](const auto& ... args) { init_time_manager<go::moves_to_go>(args...); })),
          sequential(key<int>("winc"), key<int>("binc"), invoke([&](const auto& ... args) { init_time_manager<go::increment>(args...); }))
        ))
      )),

      invoke([&] { go(); })
    )),

    sequential(consume("ponderhit"), invoke([&] { ponder_hit(); })),
    sequential(consume("stop"), invoke([&] { stop(); })),
    sequential(consume("quit"), invoke([&] { quit(); })),

    // extensions
    sequential(consume("export"), emit<std::string>, invoke([&] (const std::string& export_path) { export_weights(export_path); })),
    sequential(consume("perft"), emit<search::depth_type>, invoke([&] (const search::depth_type& depth) { perft(depth); })),
    sequential(consume("config"), invoke([&] { tune_config(); })),
    sequential(consume("bench"), invoke([&] { bench(); })),
    sequential(consume("probe"), invoke([&] { probe(); })),
    sequential(consume("eval"), invoke([&] { eval(); }))
  );

  // clang-format on

  const auto lexed = command_lexer{}.lex(line);
  processor.process(lexed.view(), {});
}

uci::uci() noexcept
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
  nnue::embedded_weight_streamer embedded(nnue::embed::weights_file_data);
  weights_.load(embedded);
  orchestrator_.resize(default_thread_count);
}

}  // namespace engine