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
#include <move.h>
#include <move_picker.h>
#include <transposition_table.h>



namespace chess{

template<typename T>
inline constexpr T eta = static_cast<T>(0.00001);

template<typename T>
inline constexpr T big_number = static_cast<T>(256.0);

template<typename T>
inline constexpr T mate_score = -std::numeric_limits<T>::max();

template<typename T>
inline constexpr T draw_score = static_cast<T>(0);

template<typename T, bool is_root> struct pvs_result{};
template<typename T> struct pvs_result<T, false>{ using type = T; };
template<typename T> struct pvs_result<T, true>{ using type = std::tuple<T, move>; };

template<typename T, bool is_root>
using pvs_result_t = typename pvs_result<T, is_root>::type;

template<typename T>
struct thread_worker{
  using real_t = T;
  static constexpr int score_num_bytes = sizeof(std::uint32_t);
  static_assert(score_num_bytes == sizeof(T), "4 byte wide T required");

  std::mutex go_mutex_{};
  std::condition_variable cv_{};
  std::atomic<bool> go_{false};
  std::atomic<int> depth_;

  std::atomic<std::uint32_t> score_;
  std::atomic<std::uint32_t> best_move_{};

  std::shared_ptr<table> tt_;
  nnue::half_kp_eval<T> evaluator_;

  std::mutex position_mutex_{};
  board position_{};

  template<bool is_pv, bool is_root=false>
  auto pv_search(const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T beta, int depth) -> pvs_result_t<T, is_root> {
    auto make_result = [](const T& score, const move& mv){
        if constexpr(is_root){
          return pvs_result_t<T, is_root>{score, mv};
      }else{
        return score;
      }
    };

    const auto list = bd.generate_moves();
    const auto empty_move = move{};

    const bool is_check = bd.is_check();
    if(list.size() == 0 && is_check){ return make_result(mate_score<T>, empty_move); }
    if(list.size() == 0) { return make_result(draw_score<T>, empty_move); }
  
    if(is_check){ depth += 1; }
  
    if(depth <= 0) { return make_result(eval.propagate(bd.turn()), empty_move); }

    T best_score = mate_score<T>;
    move best_move = *list.begin();

    if(const auto it = tt_ -> find(bd.hash()); it != tt_ -> end()){
      const tt_entry entry = *it;
      if(!is_pv && entry.depth() >= depth){
        if(entry.score() >= beta ? (entry.bound() == bound_type::lower) : (entry.bound() == bound_type::upper)){
          return make_result(entry.score(), entry.best_move());
        }
      }else if(list.has(entry.best_move())){
        best_move = entry.best_move();
      }
    }

    if(go_.load(std::memory_order_relaxed)){
      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(best_move, eval);
      const board bd_ = bd.forward(best_move);
      best_score = -pv_search<true, false>(eval_, bd_, -beta, -alpha, depth - 1);
      alpha = std::max(alpha, best_score);
    }

    auto picker = move_picker(list);

    while(!picker.empty() && go_.load(std::memory_order_relaxed)){
      const auto mv = picker.pick();
      if(best_score > beta){ break; }
      if(mv == best_move){ continue; }

      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
      const board bd_ = bd.forward(mv);
      T score = -pv_search<false, false>(eval_, bd_, -alpha - eta<T>, -alpha, depth - 1);

      if(score > alpha && score < beta){
        score = -pv_search<true, false>(eval_, bd_, -beta, -alpha, depth - 1);
        alpha = std::max(alpha, score);
      }

      if(score > best_score){
        best_score = score;
        best_move = mv;
      }
    }

    if(best_score > beta && go_.load(std::memory_order_relaxed)){
      const tt_entry entry(bd.hash(), bound_type::lower, best_score, best_move, depth);
      tt_ -> insert(entry);
    }else if(go_.load(std::memory_order_relaxed)){
      const tt_entry entry(bd.hash(), bound_type::upper, best_score, best_move, depth);
      tt_ -> insert(entry);
    }

    return make_result(best_score, best_move);
  }


  auto pv_search(const board bd, const int depth) -> pvs_result_t<T, true> {
    constexpr T alpha = -big_number<T>;
    constexpr T beta = big_number<T>;
    return pv_search<true, true>(evaluator_, bd, alpha, beta, depth);
  }


  void iterative_deepening_loop_(){
    for(;;){
      std::unique_lock<std::mutex> go_lk(go_mutex_);
      cv_.wait(go_lk, [this]{ return go_.load(std::memory_order_relaxed); });

      std::unique_lock<std::mutex> position_lk(position_mutex_);
      const auto bd = position_;
      position_lk.unlock();

      auto[nnue_score, mv] = pv_search(bd, depth_.load());
      std::uint32_t as_uint32; std::memcpy(&as_uint32, &nnue_score, score_num_bytes);
      score_.store(as_uint32);
      best_move_.store(mv.data);

      ++depth_;
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
    go_.store(true);
    go_lk.unlock();
    cv_.notify_one();
  }

  void stop(){
    go_.store(false, std::memory_order_relaxed);
  }


  thread_worker<T>& set_position(const board& bd){
    std::lock_guard<std::mutex> position_lk(position_mutex_);
    bd.show_init(evaluator_);
    position_ = bd;
    return *this;
  }

  thread_worker(const nnue::half_kp_weights<T>* weights, std::shared_ptr<table> tt, int start_depth=0) : tt_{tt}, evaluator_(weights){
    depth_.store(start_depth);
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
    for(size_t i(0); i < num_workers; ++i){
      const int start_depth = static_cast<int>(i % 2);
      pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_, start_depth));
    }
  }
};

}
