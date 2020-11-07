#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <chrono>

#include <version.h>
#include <board.h>
#include <move.h>
#include <thread_worker.h>
#include <option_parser.h>
#include <time_manager.h>

namespace engine{



struct uci{
  static constexpr size_t default_thread_count = 1;
  static constexpr size_t default_hash_size = 128;
  static constexpr std::string_view default_weight_path = "../train/model/save.bin";
  
  using real_t = float;

  chess::position_history history{};
  chess::board position = chess::board::start_pos();
  
  nnue::weights<real_t> weights_{};
  chess::worker_pool<real_t> pool_;

  bool go_{false};
  time_manager manager_{};

  std::ostream& os = std::cout;

  bool searching() const { return go_; }
  
  auto options(){
    auto weight_path = option_callback(string_option("Weights", std::string(default_weight_path)), [this](const std::string& path){
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
    constexpr real_t score_scale = static_cast<real_t>(400.0);
    constexpr int eval_limit = 256 * 100;
    
    const real_t raw_score = pool_.primary_worker().score();
    const int scaled_score = static_cast<int>(score_scale * raw_score);
    const int score = std::min(std::max(scaled_score, -eval_limit), eval_limit);
    
    static int last_reported_depth{0};
    
    const int depth = pool_.primary_worker().depth();
    const size_t elapsed_ms = manager_.elapsed().count();
    const size_t node_count = pool_.nodes();
    const size_t nps = static_cast<size_t>(1000) * node_count / (1 + elapsed_ms);
    
    if(last_reported_depth != depth){
      last_reported_depth = depth;
      os 
         << "info depth " << depth << " score cp " << score
         << " nodes " << node_count << " nps " << nps
         << " time " << elapsed_ms << " pv " << pool_.pv_string(position)
         << '\n';
    }
  }

  void go(const std::string& line){
    go_ = true;
    manager_.init(position.turn(), line);
    pool_.set_position(history, position);
    pool_.go();
  }

  void stop(){
    os << "bestmove " << pool_.primary_worker().best_move().name(position.turn()) << std::endl;
    pool_.stop();
    go_ = false;
  }

  void ready(){
    os << "readyok\n";
  }

  void id_info(){
    os << "id name Seer " << version::major << "." << version::minor << "\n";
    os << "id author Connor McMonigle\n";
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
      search_info info{
        pool_.primary_worker().depth(),
        pool_.primary_worker().is_stable()
      };
      
      if(manager_.should_stop(info)){ stop(); }
      else{ info_string(); }
    }
  }

  uci() : pool_(&weights_, default_hash_size, default_thread_count) {
    weights_.load(std::string(default_weight_path));
  }
};

}
