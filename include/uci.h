#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <chrono>

#include <board.h>
#include <move.h>
#include <thread_worker.h>
#include <option_parser.h>

namespace engine{



struct uci{
  static constexpr size_t default_thread_count = 1;
  static constexpr size_t default_hash_size = 128;
  static constexpr std::string_view default_weight_path = "/home/connor/Documents/GitHub/seer-nnue/train/model/save.bin";
  
  using real_t = float;

  chess::position_history history{};
  chess::board position = chess::board::start_pos();
  nnue::half_kp_weights<engine::uci::real_t> weights_{};

  chess::worker_pool<real_t> pool_;

  bool go_{false};
  std::chrono::milliseconds budget{0};
  std::chrono::steady_clock::time_point search_start{};
  std::ostream& os = std::cout;

  auto options(){
    auto weight_path = option_callback(string_option("Weights"), [this](const std::string& path){
      weights_.load(path);
    });

    auto hash_size = option_callback(spin_option("Hash", default_hash_size, spin_range{1, 65536}), [this](const int size){
      const auto new_size = static_cast<size_t>(size);
      pool_.tt_ -> resize(new_size);
    });

    auto thread_count = option_callback(spin_option("Threads", default_thread_count, spin_range{1, 512}), [this](const int count){
      const auto new_count = static_cast<size_t>(count);
      pool_.grow(new_count);
    });

    auto clear_hash = option_callback(button_option("Clear Hash"), [this](bool){
      pool_.tt_ -> clear();
    });

    return uci_options(weight_path, hash_size, thread_count, clear_hash);
  }

  void uci_new_game(){
    history.clear();
    pool_.tt_ -> clear();
    position = chess::board::start_pos();
  }

  void set_position(const std::string& line){
    if(line == "position startpos"){ uci_new_game(); return; }
    std::regex spos_w_moves("position startpos moves((?: [a-h][1-8][a-h][1-8]+(?:q|r|b|n)?)+)");
    std::regex fen_w_moves("position fen (.*) moves((?: [a-h][1-8][a-h][1-8]+(?:q|r|b|n)?)+)");
    std::regex fen("position fen (.*)");
    std::smatch matches{};
    if(std::regex_search(line, matches, spos_w_moves)){
      auto [h_, p_] = chess::board::start_pos().after_uci_moves(matches.str(1));
      history = h_; position = p_;
    }else if(std::regex_search(line, matches, fen_w_moves)){
      position = chess::board::parse_fen(matches.str(1));
      auto [h_, p_] = position.after_uci_moves(matches.str(2));
      history = h_; position = p_;
    }else if(std::regex_search(line, matches, fen)){
      history.clear();
      position = chess::board::parse_fen(matches.str(1));
    }
  }

  void info_string(){
    constexpr real_t eval_limit = static_cast<real_t>(256);
    
    const real_t raw_score = pool_.pool_[0] -> score();
    const real_t clamped_score = std::max(std::min(eval_limit, raw_score), -eval_limit);
    auto score = static_cast<int>(clamped_score * 600.0);
    static int last_reported_depth{0};
    const int depth = pool_.pool_[0] -> depth();
    const size_t elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - search_start).count();
    const size_t node_count = pool_.nodes();
    const size_t nps = static_cast<size_t>(1000) * node_count / (elapsed_ms+1);
    if(last_reported_depth != depth){
      last_reported_depth = depth;
      os << "info depth " << depth << " seldepth " << depth << " multipv 1 score cp " << score;
      os << " nodes " << node_count << " nps " << nps << " tbhits " << 0 << " time " << elapsed_ms << " pv " << pool_.pv_string(position) << '\n';
    }
  }

  void go(const std::string& line){
    go_ = true;
    std::regex go_w_time("go .*wtime ([0-9]+) .*btime ([0-9]+)");
    std::smatch matches{};
    pool_.set_position(history, position);
    pool_.go();
    if(std::regex_search(line, matches, go_w_time)){
      const long long our_time = std::stoll(position.turn() ? matches.str(1) : matches.str(2));
      //budget 1/7 remaing time
      budget = std::chrono::milliseconds(our_time / 7);
      search_start = std::chrono::steady_clock::now();
    }else{
      //this is very dumb
      budget = std::chrono::milliseconds(1ull << 32ull);
      search_start = std::chrono::steady_clock::now();
    }
  }

  void stop(){
    os << "bestmove " << pool_.pool_[0] -> best_move().name(position.turn()) << std::endl;
    pool_.stop();
    go_ = false;
  }

  void ready(){
    os << "readyok\n";
  }

  void id_info(){
    os << "id name Seer\n";
    os << "id author C. McMonigle\n";
    os << options();
    os << "uciok\n";
  }

  void uci_loop(const std::string& line){
    std::regex position_rgx("position(.*)");
    std::regex go_rgx("go(.*)");
    if(line == "uci"){
      id_info();
    }else if(line == "isready"){
      ready();
    }else if(line == "ucinewgame"){
      uci_new_game();
    }else if(line == "stop"){
      stop();
    }else if(line == "_internal_board"){
      os << position << std::endl;
    }else if(std::regex_match(line, go_rgx)){
      go(line);
    }else if(std::regex_match(line, position_rgx)){
      set_position(line);
    }else if(line == "quit"){
      std::terminate();
    }else if(!go_){
      options().update(line);
    }

    if(go_){
      if((std::chrono::steady_clock::now() - search_start) >= budget){
        stop();
      }else{
        info_string();
      }
    }
  }

  uci() : pool_(&weights_, default_hash_size, default_thread_count) {
    weights_.load(std::string(default_weight_path));
  }
};

}
