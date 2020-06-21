#pragma once

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <vector>

#include <nnue_half_kp.h>
#include <board.h>
#include <transposition_table.h>
#include <search.h>

namespace chess{

template<typename T>
struct thread_worker{
  static constexpr int score_num_bytes = sizeof(std::uint32_t);
  static_assert(score_num_bytes == sizeof(T), "4 byte wide T required");

  std::mutex go_mutex_{};
  std::condition_variable cv_{};
  bool go_{false};
  std::atomic<int> depth_{0};

  std::atomic<std::uint32_t> score_;
  std::atomic<std::uint32_t> best_move_{};

  std::shared_ptr<table> tt_;
  nnue::half_kp_eval<T> evaluator_;

  std::mutex position_mutex_{};
  board position_{};

  void iterative_deepening_loop_(){
    for(;;){
      std::unique_lock<std::mutex> go_lk(go_mutex_);
      cv_.wait(go_lk, [this]{ return go_; });

      std::unique_lock<std::mutex> position_lk(position_mutex_);
      const auto bd = position_;
      position_lk.unlock();

      auto[nnue_score, mv] = pv_search(tt_, evaluator_, bd, depth_);
      std::uint32_t as_uint32; std::memcpy(&as_uint32, &nnue_score, score_num_bytes);
      score_.store(as_uint32);
      best_move_.store(mv.data);

      depth_.store(depth_.load() + 1);
    }
  }

  int depth() const {
    return depth_.load();
  }

  move best_move() const {
    return move{best_move_.load()};
  }

  float score() const {
    const std::uint32_t raw = score_.load();
    float result; std::memcpy(&result, &raw, score_num_bytes);
    return result;
  }

  void go(){
    std::unique_lock<std::mutex> go_lk(go_mutex_);
    depth_.store(0);
    go_ = true;
    go_lk.unlock();
    cv_.notify_one();
  }

  void stop(){
    std::unique_lock<std::mutex> go_lk(go_mutex_);
    go_ = false;
    go_lk.unlock();
    cv_.notify_one();
  }


  thread_worker<T>& set_position(const board& bd){
    std::lock_guard<std::mutex> position_lk(position_mutex_);
    bd.show_init(evaluator_);
    position_ = bd;
    return *this;
  }

  thread_worker(const nnue::half_kp_weights<T>* weights, std::shared_ptr<table> tt) : tt_{tt}, evaluator_(weights){
    std::thread([this]{ iterative_deepening_loop_(); }).detach();
  }
};

template<typename T>
struct worker_pool{
  std::shared_ptr<table> tt_{nullptr};
  std::vector<std::shared_ptr<thread_worker<T>>> pool_{};

  void go(){
    for(auto& worker : pool_){ worker -> go(); }
  }
  
  void stop(){
    for(auto& worker : pool_){ worker -> stop(); }
  }

  void set_position(const board& bd){
    for(auto& worker : pool_){ worker -> set_position(bd); }
  }

  worker_pool(const nnue::half_kp_weights<T>* weights, size_t hash_table_size, size_t num_workers){
    tt_ = std::make_shared<table>(hash_table_size);
    for(size_t i(0); i < num_workers; ++i){ pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_)); }
  }
};

}
