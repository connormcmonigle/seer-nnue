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
#include <history_heuristic.h>


namespace chess{

template<typename T>
inline constexpr T eta = static_cast<T>(0.0001);

template<typename T>
inline constexpr T big_number = static_cast<T>(256);

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
  std::atomic<size_t> nodes_;

  std::atomic<std::uint32_t> score_;
  std::atomic<std::uint32_t> best_move_{};

  nnue::half_kp_eval<T> evaluator_;
  std::shared_ptr<table> tt_;
  std::shared_ptr<sided_history_heuristic> hh_;

  std::mutex position_mutex_{};
  board position_{};
  position_history history_{};

  auto quiescent_search(position_history& hist, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T beta) -> T const {
    ++nodes_;
    
    const auto list = bd.generate_moves();

    const bool is_check = bd.is_check();
    if(list.size() == 0 && is_check){ return mate_score<T>; }
    if(list.size() == 0) { return draw_score<T>; }
    if(hist.is_three_fold(bd.hash())){ return draw_score<T>; }

    auto best_score = mate_score<T>;
    const std::optional<tt_entry> entry = tt_ -> find(bd.hash());
    
    if(entry.has_value()){
      if(entry.value().score() >= beta && entry.value().bound() == bound_type::lower){
        return entry.value().score();
      }
      alpha = std::max(alpha, entry.value().score());
    }

    const T static_eval = eval.propagate(bd.turn());
    const auto loud_list = list.loud();

    if(loud_list.size() == 0){ return static_eval; }
    if(static_eval > beta){ return static_eval; }

    const int pre_size = hist.history_.size();
    hist.push_(bd.hash());

    alpha = std::max(alpha, static_eval);    
    best_score = std::max(best_score, static_eval);

    auto picker = move_picker(loud_list, &(hh_ -> us(bd.turn())));
    const auto first_move = entry.has_value() ? entry.value().best_move() : picker.peek();

    if(go_.load(std::memory_order_relaxed)){
      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(first_move, eval);
      const board bd_ = bd.forward(first_move);
      const T score = -quiescent_search(hist, eval_, bd_, -beta, -alpha);
      alpha = std::max(alpha, score);
      best_score = std::max(best_score, score);
    }

    while(!picker.empty() && go_.load(std::memory_order_relaxed)){
      const auto mv = picker.pick();
      if(best_score > beta){ break; }
      if(mv == first_move){ continue; }

      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
      const board bd_ = bd.forward(mv);
      

      const T score = -quiescent_search(hist, eval_, bd_, -beta, -alpha);
      alpha = std::max(alpha, score);
      best_score = std::max(best_score, score);
    }

    hist.pop_();
    const int post_size = hist.history_.size();
    assert(post_size == pre_size);

    return best_score;
  }

  template<bool is_pv, bool is_root=false>
  auto pv_search(position_history& hist, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T beta, int depth) -> pvs_result_t<T, is_root> const {
    ++nodes_;
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
    if(hist.is_three_fold(bd.hash())){ return make_result(draw_score<T>, empty_move); }
    
    if(is_check){ depth += 1; }
  
    if(depth <= 0) { return make_result(quiescent_search(hist, eval, bd, alpha, beta), empty_move); }

    auto picker = move_picker(list, &(hh_ -> us(bd.turn())));
    move first_move = picker.peek();

    if(const std::optional<tt_entry> entry = tt_ -> find(bd.hash()); entry.has_value()){
      if(!is_pv && entry.value().depth() >= depth){
        if(entry.value().score() >= beta ? (entry.value().bound() == bound_type::lower) : (entry.value().bound() == bound_type::upper)){
          return make_result(entry.value().score(), entry.value().best_move());
        }
      }else if(list.has(entry.value().best_move())){
        first_move = entry.value().best_move();
      }
    }
    
    const int pre_size = hist.history_.size();
    hist.push_(bd.hash());

    T best_score = mate_score<T>;
    move best_move = first_move;

    if(go_.load(std::memory_order_relaxed)){
      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(best_move, eval);
      const board bd_ = bd.forward(best_move);
      best_score = -pv_search<true, false>(hist, eval_, bd_, -beta, -alpha, depth - 1);
      alpha = std::max(alpha, best_score);
    }

    while(!picker.empty() && go_.load(std::memory_order_relaxed)){
      const auto mv = picker.pick();
      if(best_score > beta){ break; }
      if(mv == first_move){ continue; }

      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
      const board bd_ = bd.forward(mv);
      
      T score = -pv_search<false, false>(hist, eval_, bd_, -alpha - eta<T>, -alpha, depth - 1);

      if(score > alpha && score < beta){
        score = -pv_search<true, false>(hist, eval_, bd_, -beta, -alpha, depth - 1);
        alpha = std::max(alpha, score);
      }

      if(score > best_score){
        best_score = score;
        best_move = mv;
      }
    }

    if(go_.load(std::memory_order_relaxed)){
      if(best_score > beta){
        const tt_entry entry(bd.hash(), bound_type::lower, best_score, best_move, depth);
        tt_-> insert(entry);
        (hh_ -> us(bd.turn())).add(depth, best_move);
      }else{
        const tt_entry entry(bd.hash(), bound_type::upper, best_score, best_move, depth);
        tt_-> insert(entry);
      }
    }
    
    hist.pop_();
    const int post_size = hist.history_.size();
    assert(post_size == pre_size);

    return make_result(best_score, best_move);
  }



  auto root_search(position_history& hist, const board bd, const int depth) -> pvs_result_t<T, true> const {
    constexpr T alpha = -big_number<T>;
    constexpr T beta = big_number<T>;
    return pv_search<true, true>(hist, evaluator_, bd, alpha, beta, depth);
  }


  void iterative_deepening_loop_(){
    for(;;){
      std::unique_lock<std::mutex> go_lk(go_mutex_);
      cv_.wait(go_lk, [this]{ return go_.load(std::memory_order_relaxed); });

      std::unique_lock<std::mutex> position_lk(position_mutex_);
      const auto bd = position_;
      auto hist = history_;
      position_lk.unlock();

      //increment table generation on every new root search
      tt_ -> update_gen();
      auto[nnue_score, mv] = root_search(hist, bd, depth_.load());

      std::uint32_t as_uint32; std::memcpy(&as_uint32, &nnue_score, score_num_bytes);
      score_.store(as_uint32);
      best_move_.store(mv.data);

      ++depth_;
    }
  }

  int depth() const {
    return depth_.load();
  }

  size_t nodes() const {
    return nodes_.load();
  }

  move best_move() const {
    return move{best_move_.load()};
  }

  float score() const {
    const std::uint32_t raw = score_.load();
    float result; std::memcpy(&result, &raw, score_num_bytes);
    return result;
  }

  void go(const int start_depth){
    std::unique_lock<std::mutex> go_lk(go_mutex_);
    depth_.store(start_depth);
    nodes_.store(0);
    go_.store(true);
    go_lk.unlock();
    cv_.notify_one();
  }

  void stop(){
    go_.store(false, std::memory_order_relaxed);
  }


  thread_worker<T>& set_position(const position_history& hist, const board& bd){
    std::lock_guard<std::mutex> position_lk(position_mutex_);
    bd.show_init(evaluator_);
    history_ = hist;
    position_ = bd;
    return *this;
  }

  thread_worker(const nnue::half_kp_weights<T>* weights, std::shared_ptr<table> tt, std::shared_ptr<sided_history_heuristic> hh) : 
    evaluator_(weights), tt_{tt}, hh_{hh} {
    std::thread([this]{ iterative_deepening_loop_(); }).detach();
  }
};

template<typename T>
struct worker_pool{
  
  const nnue::half_kp_weights<T>* weights_;
  std::shared_ptr<table> tt_{nullptr};
  std::shared_ptr<sided_history_heuristic> hh_{nullptr};

  std::vector<std::shared_ptr<thread_worker<T>>> pool_{};

  std::string pv_string(chess::board bd) const {
    std::string result{};
    constexpr size_t max_pv_length = 256;
    for(size_t i(0); i < max_pv_length; ++i){
      if(const std::optional<tt_entry> entry = tt_ -> find(bd.hash()); entry.has_value()){
        if(bd.generate_moves().has(entry.value().best_move())){
          result += entry.value().best_move().name(bd.turn()) + " ";
          bd = bd.forward(entry.value().best_move());
        }else{ break; }
      }else{ break; }
    }
    return result;
  }

  void grow(size_t new_size){
    assert((new_size > pool_.size()));
    const size_t new_workers = new_size - pool_.size();
    for(size_t i(0); i < new_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights_, tt_, hh_));
    }
  }

  void go(){
    for(size_t i(0); i < pool_.size(); ++i){
      const int start_depth = static_cast<int>(i % 2);
      pool_[i] -> go(start_depth);
    }
  }
  
  void stop(){
    for(auto& worker : pool_){ worker -> stop(); }
  }

  size_t nodes() const {
    return std::accumulate(pool_.cbegin(), pool_.cend(), static_cast<size_t>(0), [](size_t count, const auto& worker){
      return count + worker -> nodes();
    });
  }

  void set_position(const position_history& hist, const board& bd){
    for(auto& worker : pool_){ worker -> set_position(hist, bd); }
  }

  worker_pool(const nnue::half_kp_weights<T>* weights, size_t hash_table_size, size_t num_workers) : weights_{weights} {
    tt_ = std::make_shared<table>(hash_table_size);
    hh_ = std::make_shared<sided_history_heuristic>();
    for(size_t i(0); i < num_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_, hh_));
    }
  }
};

}
