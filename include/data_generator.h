#pragma once
#include <fstream>
#include <random>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>

#include <sample.h>
#include <move.h>
#include <transposition_table.h>
#include <search_util.h>
#include <thread_worker.h>

namespace selfplay{

namespace config_hardcode{
  
constexpr size_t tt_mb_size = 2;
constexpr search::depth_type move_limit = 512;

constexpr search::depth_type high_temperature_count = 6;
constexpr search::depth_type low_temperature_count = 20;
constexpr double high_temperature = 0.08;
constexpr double low_temperature = 0.008;

}

struct default_random_inserter{  
  std::mt19937 mt;
  
  std::bernoulli_distribution high_temp;
  std::bernoulli_distribution low_temp;
  
  bool make_random_(const search::depth_type& move_count){
    return
      ((move_count <= config_hardcode::high_temperature_count) && high_temp(mt)) || 
      ((config_hardcode::high_temperature_count < move_count && move_count <= config_hardcode::low_temperature_count) && low_temp(mt));
  }
  
  chess::move sample(const search::depth_type move_count, const chess::move& best, const chess::move_list& ls){
    assert(ls.size() != 0);
    if(ls.size() == 1){ return best; }
    std::uniform_int_distribution<size_t> index_dist(0, ls.size() - 1);
    return make_random_(move_count) ? best : ls.data[index_dist(mt)];
  }
  
  template<typename S>
  default_random_inserter(const S& seed) : mt{seed}, high_temp{config_hardcode::high_temperature}, low_temp{config_hardcode::low_temperature} {}
};

template<typename T, typename R>
struct generator_config{
  const nnue::weights<T>* weights_{nullptr};
  search::depth_type depth_;
  R random_inserter_;
  
  std::shared_ptr<chess::table> tt_{nullptr};
  std::shared_ptr<search::constants> constants_{nullptr};
  
  
  generator_config<T, R> clone() const {
    return generator_config<T, R>(weights_, depth_);
  }
  
  generator_config(const nnue::weights<T>* weights, const search::depth_type& depth) : weights_{weights}, depth_{depth},  random_inserter_(std::random_device()()) {
      tt_ = std::make_shared<chess::table>(config_hardcode::tt_mb_size);
      constants_ = std::make_shared<search::constants>(1);
  }
};

template<typename T, typename R>
struct generator_worker{
  generator_config<T, R> cfg_;

  chess::thread_worker<T> instance;
  
  std::mutex buffer_mutex;
  std::vector<sample> buffer{};
  
  
  std::vector<sample> generate_single_game(){
    cfg_.tt_ -> clear();
    chess::position_history history{};
    std::vector<sample> game_buffer{};
    auto state = chess::board::start_pos();
    // gross
    for(;;){
      if(state.generate_moves().size() == 0){ break; }
      if(state.lat_.move_count >= config_hardcode::move_limit){ break; }

      instance.set_position(history, state);
      instance.go(1);
      while(instance.depth() < cfg_.depth_){ std::this_thread::sleep_for(std::chrono::microseconds(200)); }
      instance.stop();
      
      game_buffer.push_back(sample{state, instance.score(), 0});
      history.push_(state.hash());
      
      assert((state.generate_moves().has(instance.best_move())));
      state = state.forward(
        cfg_.random_inserter_.sample(
          state.lat_.move_count,
          instance.best_move(),
          state.generate_moves()
        )
      );
      
      if(instance.score() >= -search::max_mate_score){ break; }
      if(instance.score() <= search::max_mate_score){ break; }

    }
    
    int outcome = [&, this]{
      if(state.generate_moves().size() == 0 && state.is_check()){ return -1; }
      if(instance.score() <= search::max_mate_score){ return -1; }
      if(instance.score() >= -search::max_mate_score){ return 1; }
      return 0;
    }();
    
    std::for_each(game_buffer.rbegin(), game_buffer.rend(), [&outcome](sample& s){
      s.outcome = outcome;
      outcome = -outcome;
    });
    
    return game_buffer;
  }
  
  void generate_n_positions(const size_t& n){
    size_t count{0};
    while(count < n){
      const std::vector<sample> game = generate_single_game();
      std::lock_guard<std::mutex> lock(buffer_mutex);
      if((count + game.size()) >= n){
        const size_t remainder = n - count;
        buffer.insert(buffer.end(), game.begin(), game.begin() + remainder);
        count += remainder;
      }else{
        buffer.insert(buffer.end(), game.begin(), game.end());
        count += game.size();
      }
    }
    assert(n == count);
  }
  
  generator_worker(generator_config<T, R> cfg, const size_t& n) : 
    cfg_{cfg},
    instance(cfg.weights_, cfg.tt_, cfg.constants_) 
  {
    std::thread([this, n]{ generate_n_positions(n); }).detach();
  }

};

template<typename T, typename R>
struct generator{
  size_t total_written{0};
  std::vector<std::shared_ptr<generator_worker<T, R>>> workers{};
  
  void poll(std::fstream& file){
    for(auto& worker : workers){
      std::lock_guard<std::mutex>(worker -> buffer_mutex);
      total_written += (worker -> buffer).size();
      std::for_each((worker -> buffer).begin(), (worker -> buffer).end(), [&file](const sample& s){
        file << s << '\n';
      });
      (worker -> buffer).clear();
    }
  }
  
  generator(generator_config<T, R> cfg, const size_t& num_workers, const size_t& num_positions){
    assert((0 == num_positions % num_workers));
    for(size_t i(0); i < num_workers; ++i){
      workers.push_back(std::make_shared<generator_worker<T, R>>(cfg.clone(), num_positions / num_workers));
    }
  }
};

  
}
