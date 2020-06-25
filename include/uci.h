#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <regex>
#include <chrono>

#include <board.h>
#include <move.h>
#include <thread_worker.h>

namespace engine{

struct uci{
  using real_t = float;
  chess::board position = chess::board::start_pos();
  chess::worker_pool<real_t> pool_;

  bool go_{false};
  std::chrono::milliseconds budget{0};
  std::chrono::steady_clock::time_point search_start{};
  std::ostream& os = std::cout;

  void uci_new_game(){
    position = chess::board::start_pos();
  }

  void set_position(const std::string& line){
    if(line == "position startpos"){ uci_new_game(); return; }
    std::regex spos_w_moves("position startpos moves((?: [a-h][1-8][a-h][1-8])+)");
    std::regex fen_w_moves("position fen (.*) moves((?: [a-h][1-8][a-h][1-8])+)");
    std::regex fen("position fen (.*)");
    std::smatch matches{};
    if(std::regex_search(line, matches, spos_w_moves)){
      position = chess::board::start_pos().after_uci_moves(matches.str(1));
    }else if(std::regex_search(line, matches, fen_w_moves)){
      position = chess::board::parse_fen(matches.str(1));
      position = position.after_uci_moves(matches.str(2));
    }else if(std::regex_search(line, matches, fen)){
      position = chess::board::parse_fen(matches.str(1));
    }
  }

  void info_string(){
    const real_t raw_score = pool_.pool_[0] -> score();
    const real_t clamped_score = std::max(std::min(chess::big_number<real_t>, raw_score), -chess::big_number<real_t>);
    auto score = static_cast<int>(clamped_score * 80.0);
    static int last_reported_depth{0};
    const int depth = pool_.pool_[0] -> depth();
    if(last_reported_depth != depth){
      last_reported_depth = depth;
      os << "info depth " << depth << " seldepth " << depth << " multipv 1 score cp " << score << '\n';
    }
  }

  void go(const std::string& line){
    go_ = true;
    std::regex go_w_time("go .*wtime ([0-9]+) .*btime ([0-9]+)");
    std::smatch matches{};
    pool_.set_position(position);
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
    go_ = false;
    os << "bestmove " << pool_.pool_[0] -> best_move().name(position.turn()) << std::endl;
    pool_.stop();
  }

  void ready(){
    os << "readyok\n";
  }

  void id_info(){
    os << "id name seer-nnue-testing\n";
    os << "id author C. McMonigle\n";
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
    }
    if(go_){
      if((std::chrono::steady_clock::now() - search_start) >= budget){
        stop();
      }else{
        info_string();
      }
    }
  }

  uci(const nnue::half_kp_weights<real_t>* weights, size_t hash_size, const size_t thread_count) : pool_(weights, hash_size, thread_count) {}
};

}