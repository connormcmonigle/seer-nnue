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

#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <iterator>
#include <regex>
#include <chrono>
#include <cstdlib>
#include <atomic>
#include <thread>

#include <version.h>
#include <board.h>
#include <move.h>
#include <search_constants.h>
#include <search_stack.h>
#include <thread_worker.h>
#include <option_parser.h>
#include <time_manager.h>
#include <embedded_weights.h>
#include <bench.h>

namespace engine{
  
struct uci{
  using weight_type = float;
  
  static constexpr size_t default_thread_count = 1;
  static constexpr size_t default_hash_size = 16;
  static constexpr std::string_view default_weight_path = "EMBEDDED";


  chess::position_history history{};
  chess::board position = chess::board::start_pos();
  
  nnue::weights<weight_type> weights_{};
  chess::worker_pool<weight_type> pool_;

  std::atomic_bool should_quit_{false};
  std::atomic_bool is_searching_{false};

  time_manager manager_;
  simple_timer<std::chrono::milliseconds> timer_;

  std::mutex os_mutex_{}; 
  std::ostream& os = std::cout;

  bool should_quit() const { return should_quit_.load(); }
  bool is_searching() const { return is_searching_.load(); }

  void weights_info_string(){
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << "info string loaded weights with signature 0x" << std::hex << weights_.signature() << std::dec << std::endl;
  }

  auto options(){
    auto weight_path = option_callback(string_option("Weights", std::string(default_weight_path)), [this](const std::string& path){
      if(path == std::string(default_weight_path)){
        nnue::embedded_weight_streamer<weight_type> embedded(embed::weights_file_data);
        weights_.load(embedded);
      }else{
        weights_.load(path);
      }
      weights_info_string();
    });

    auto hash_size = option_callback(spin_option("Hash", default_hash_size, spin_range{1, 65536}), [this](const int size){
      const auto new_size = static_cast<size_t>(size);
      pool_.tt_ -> resize(new_size);
    });

    auto thread_count = option_callback(spin_option("Threads", default_thread_count, spin_range{1, 512}), [this](const int count){
      const auto new_count = static_cast<size_t>(count);
      pool_.resize(new_count);
    });

    return uci_options(weight_path, hash_size, thread_count);
  }
  
  void uci_new_game(){
    history.clear();
    pool_.tt_ -> clear();
    position = chess::board::start_pos();
  }

  void set_position(const std::string& line){
    const std::regex startpos_with_moves("position startpos moves((?: [a-h][1-8][a-h][1-8]+(?:q|r|b|n)?)+)");
    const std::regex fen_with_moves("position fen (.*) moves((?: [a-h][1-8][a-h][1-8]+(?:q|r|b|n)?)+)");
    const std::regex startpos("position startpos");
    const std::regex fen("position fen (.*)");
    std::smatch matches{};
    
    if(std::regex_search(line, matches, startpos_with_moves)){
      auto [h_, p_] = chess::board::start_pos().after_uci_moves(matches.str(1));
      history = h_; position = p_;
    }else if(std::regex_search(line, matches, fen_with_moves)){
      position = chess::board::parse_fen(matches.str(1));
      auto [h_, p_] = position.after_uci_moves(matches.str(2));
      history = h_; position = p_;
    }else if(std::regex_search(line, matches, startpos)){
      history.clear();
      position = chess::board::start_pos();
    }else if(std::regex_search(line, matches, fen)){
      history.clear();
      position = chess::board::parse_fen(matches.str(1));
    }
  }

  template<typename T>
  void info_string(const T& worker){
    constexpr search::score_type raw_multiplier = 400;
    constexpr search::score_type raw_divisor = 1024;
    constexpr search::score_type eval_limit = 256 * 100;
    
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    
    const search::score_type raw_score = worker.score();
    const search::score_type scaled_score = raw_score * raw_multiplier / raw_divisor;
    const int score = std::min(std::max(scaled_score, -eval_limit), eval_limit);
    
    
    const int depth = worker.depth();
    const size_t elapsed_ms = timer_.elapsed().count();
    const size_t nodes = pool_.nodes();
    const size_t nps = std::chrono::milliseconds(std::chrono::seconds(1)).count() * nodes / (1 + elapsed_ms);
    if(is_searching()){
      os << "info depth " << depth
         << " seldepth " << worker.internal.stack.sel_depth()
         << " score cp " << score
         << " nodes " << nodes 
         << " nps " << nps
         << " time " << elapsed_ms
         << " pv " << worker.internal.stack.pv_string()
         << std::endl;
    }
  }

  void go(const std::string& line){
    is_searching_.store(true);
    manager_.init(position.turn(), line);
    timer_.lap();
    pool_.go(history, position);
  }

  void stop(){
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    is_searching_.store(false);
    pool_.stop();
    os << "bestmove " << pool_.primary_worker().best_move().name(position.turn()) << std::endl;
  }

  void ready(){
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << "readyok" << std::endl;
  }

  void id_info(){
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << "id name " << version::engine_name << " " << version::major << '.' << version::minor << '.' << version::patch << std::endl;
    os << "id author " << version::author_name << std::endl;
    os << options();
    if constexpr(search::constants::tuning){ os << (pool_.constants_ -> options()); }
    os << "uciok" << std::endl;
  }

  void bench(){
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    os << get_bench_info(weights_) << std::endl;
  }

  void quit(){ should_quit_.store(true); }

  void read(const std::string& line){
    const std::regex position_rgx("position(.*)");
    const std::regex go_rgx("go(.*)");
    
    if(!is_searching() && line == "uci"){
      id_info();
    }else if(line == "isready"){
      ready();
    }else if(!is_searching() && line == "ucinewgame"){
      uci_new_game();
    }else if(is_searching() && line == "stop"){
      stop();
    }else if(line == "_internal_board"){
      os << position << std::endl;
    }else if(!is_searching() && line == "bench"){
      bench();
    }else if(!is_searching() && std::regex_match(line, go_rgx)){
      go(line);
    }else if(!is_searching() && std::regex_match(line, position_rgx)){
      set_position(line);
    }else if(line == "quit"){
      quit();
    }else if(!is_searching()){
      options().update(line);
      if constexpr(search::constants::tuning){ (pool_.constants_ -> options()).update(line); }
    }
  }

  void update(){
    const search_info info{pool_.primary_worker().depth(), pool_.primary_worker().is_stable()};
    if(is_searching() && manager_.should_stop(info)){ stop(); }
  }

  uci() : 
    pool_(&weights_, default_hash_size, [this](auto&&... args){ info_string(args...); })
  {
    nnue::embedded_weight_streamer<weight_type> embedded(embed::weights_file_data);
    weights_.load(embedded);
    pool_.resize(default_thread_count);
  }
};

template<typename U>
struct command_loop{
  std::mutex command_mutex_;
  std::atomic_bool running_;
  std::thread thread_;
  
  template<typename T>
  command_loop(U& u, const T& interval) : 
    running_{true},  
    thread_([&u, interval, this]{
      while(running_.load() && !u.should_quit()){
        std::this_thread::sleep_for(interval);
        std::lock_guard<std::mutex> lk(command_mutex_); 
        u.update();
      }
    })
  {
    std::string line{};
    while(running_.load() && !u.should_quit() && std::getline(std::cin, line)){
      std::lock_guard<std::mutex> lk(command_mutex_); 
      u.read(line);
    }
  }

  ~command_loop(){
    running_.store(false);
    thread_.join();
  }

};

}
