#pragma once
#include <iostream>
#include <sstream>
#include <string>
#include <string_view>
#include <iterator>
#include <regex>
#include <chrono>
#include <cstdint>
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

  std::atomic<bool> go_{false};
  simple_timer<std::chrono::milliseconds> timer_{};
  
  std::mutex os_mutex_{}; 
  std::ostream& os = std::cout;

  bool searching() const { return go_; }
  
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
    if(go_.load()){
      os << "info depth " << depth
         << " seldepth " << worker.stack_.sel_depth()
         << " score cp " << score
         << " nodes " << nodes 
         << " nps " << nps
         << " time " << elapsed_ms
         << " pv " << worker.stack_.pv_string()
         << std::endl;
    }
  }

  void go(const std::string& line){
    go_.store(true);
    pool_.set_position(history, position);
    pool_.go();
    timer_.lap();
    
    // manage time using a separate, low utilization thread
    std::thread([line, this]{
      auto manager = time_manager{}.init(position.turn(), line);
      
      while(go_.load()){
        search_info info{
          pool_.primary_worker().depth(),
          pool_.primary_worker().is_stable()
        };
      
        if(manager.should_stop(info)){
          stop();
          break;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }).detach();
  }

  void stop(){
    std::lock_guard<std::mutex> os_lk(os_mutex_);
    go_.store(false);
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

  void uci_loop(const std::string& line){
    const std::regex position_rgx("position(.*)");
    const std::regex go_rgx("go(.*)");
    
    if(!go_ && line == "uci"){
      id_info();
    }else if(line == "isready"){
      ready();
    }else if(!go_.load() && line == "ucinewgame"){
      uci_new_game();
    }else if(go_.load() && line == "stop"){
      stop();
    }else if(line == "_internal_board"){
      os << position << std::endl;
    }else if(!go_.load() && line == "bench"){
      bench();
    }else if(!go_.load() && std::regex_match(line, go_rgx)){
      go(line);
    }else if(!go_.load() && std::regex_match(line, position_rgx)){
      set_position(line);
    }else if(line == "quit"){
      std::exit(0);
    }else if(!go_.load()){
      options().update(line);
      if constexpr(search::constants::tuning){ (pool_.constants_ -> options()).update(line); }
    }
  }

  uci() : pool_(&weights_, default_hash_size, default_thread_count, [this](auto&&... args){ info_string(args...); }) {
    nnue::embedded_weight_streamer<weight_type> embedded(embed::weights_file_data);
    weights_.load(embedded);
  }
};

}
