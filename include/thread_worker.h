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
#include <search_util.h>
#include <move_orderer.h>
#include <transposition_table.h>
#include <history_heuristic.h>


namespace chess{

template<typename T>
inline constexpr T epsilon = static_cast<T>(1e-5);

template<typename T>
inline constexpr T big_number = static_cast<T>(256);

template<typename T>
inline constexpr T mate_score = -static_cast<T>(2) * big_number<T>;

template<typename T>
inline constexpr T draw_score = static_cast<T>(0);

template<typename T>
inline constexpr T aspiration_delta = static_cast<T>(0.03);

template<typename T, bool is_root> struct pv_search_result{};
template<typename T> struct pv_search_result<T, false>{ using type = T; };
template<typename T> struct pv_search_result<T, true>{ using type = std::tuple<T, move>; };

template<typename T, bool is_root>
using pv_search_result_t = typename pv_search_result<T, is_root>::type;

template<typename T>
struct thread_worker{
  using real_t = T;
  static constexpr int score_num_bytes = sizeof(std::uint32_t);
  static_assert(score_num_bytes == sizeof(T), "4 byte wide T required");

  std::mutex go_mutex_{};
  std::condition_variable cv_{};
  std::atomic<bool> go_{false};
  std::atomic<search::depth_type> depth_;
  std::atomic<size_t> nodes_;

  std::atomic<std::uint32_t> score_;
  std::atomic<std::uint32_t> best_move_{};

  nnue::half_kp_eval<T> evaluator_;
  std::shared_ptr<table> tt_;
  std::shared_ptr<sided_history_heuristic> hh_;
  std::shared_ptr<search::constants> constants_;

  std::mutex position_mutex_{};
  board position_{};
  position_history history_{};

  auto q_search(position_history& hist, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T& beta) -> T const {
    ++nodes_;
    
    const auto list = bd.generate_moves();

    const bool is_check = bd.is_check();
    if(list.size() == 0 && is_check){ return mate_score<T>; }
    if(list.size() == 0) { return draw_score<T>; }
    if(hist.is_three_fold(bd.hash())){ return draw_score<T>; }
    
    const auto loud_list = list.loud();
    auto orderer = move_orderer(&bd, loud_list, &(hh_ -> us(bd.turn())));
    
    if(const std::optional<tt_entry> maybe = tt_ -> find(bd.hash()); maybe.has_value()){
      const tt_entry entry = maybe.value();
      const bool is_cutoff = entry.score() >= beta && entry.bound() == bound_type::lower;
      if(is_cutoff){ return entry.score(); }
      alpha = std::max(alpha, entry.score());
      orderer.set_first(entry.best_move());
    }

    const T static_eval = eval.propagate(bd.turn());
    if(loud_list.size() == 0 || static_eval > beta){ return static_eval; }

    alpha = std::max(alpha, static_eval);
    T best_score = static_eval;
    
    auto _ = hist.scoped_push_(bd.hash());
    for(auto [idx, mv] : orderer){
      assert((mv != move::null()));
      if(!go_.load(std::memory_order_relaxed) || best_score > beta){ break; }
      if(bd.see<int>(mv) >= 0){
        const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
        const board bd_ = bd.forward(mv);
      
        const T score = -q_search(hist, eval_, bd_, -beta, -alpha);
        alpha = std::max(alpha, score);
        best_score = std::max(best_score, score);
      }
    }

    return best_score;
  }

  template<bool is_pv, bool is_root=false>
  auto pv_search(position_history& hist, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T& beta, search::depth_type depth) -> pv_search_result_t<T, is_root> const {
    auto make_result = [](const T& score, const move& mv){
      if constexpr(is_root){ return pv_search_result_t<T, is_root>{score, mv}; }
      if constexpr(!is_root){ return score; }
    };

    const auto list = bd.generate_moves();

    const bool is_check = bd.is_check();
    if(list.size() == 0 && is_check){ return make_result(mate_score<T>, move::null()); }
    if(list.size() == 0) { return make_result(draw_score<T>, move::null()); }
    if(hist.is_three_fold(bd.hash())){ return make_result(draw_score<T>, move::null()); }
    
    if(is_check){ depth += 1; }
  
    if(depth <= 0) { return make_result(q_search(hist, eval, bd, alpha, beta), move::null()); }
    ++nodes_;

    auto orderer = move_orderer(&bd, list, &(hh_ -> us(bd.turn())));

    if(const std::optional<tt_entry> maybe = tt_ -> find(bd.hash()); maybe.has_value()){
      const tt_entry entry = maybe.value();
      const bool is_cutoff = !is_pv &&
        entry.depth() >= depth &&
        (entry.score() >= beta ?
          (entry.bound() == bound_type::lower) :
          (entry.bound() == bound_type::upper));
      if(is_cutoff){ return make_result(entry.score(), entry.best_move()); }
      orderer.set_first(entry.best_move());
    }

    T best_score = mate_score<T>;
    move best_move = list.data[0];

    auto _ = hist.scoped_push_(bd.hash());
    for(auto [idx, mv] : orderer){
      assert((mv != move::null()));
      if(!go_.load(std::memory_order_relaxed) || best_score > beta){ break; }

      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
      const board bd_ = bd.forward(mv);
      
      const T score = [&, this]{
        const search::depth_type next_depth = depth - 1;
        auto full_width = [&]{ return -pv_search<is_pv>(hist, eval_, bd_, -beta, -alpha, next_depth); };
          
        const bool lmr = !is_check && (depth >= constants_ -> reduce_depth());
        T zw_score{};

        if(lmr){
          search::depth_type reduction = constants_ -> reduction<is_pv>(depth, idx);
          
          if(bd.forward(mv).is_check()){ --reduction; }
          if(mv.is_promotion()){ --reduction; }
          if(bd.see<int>(mv) < 0){ ++reduction; }
          
          reduction = std::max(reduction, 0);
          
          const search::depth_type lmr_depth = std::max(0, next_depth - reduction);
          zw_score = -pv_search<false>(hist, eval_, bd_, -alpha - epsilon<T>, -alpha, lmr_depth);
        }

        if(!lmr || (lmr && (zw_score > alpha))){
          zw_score = -pv_search<false>(hist, eval_, bd_, -alpha - epsilon<T>, -alpha, next_depth);
        }

        const bool interior = zw_score > alpha && zw_score < beta;
        return (interior && is_pv) ? full_width() : zw_score;
      }();

      if(score < beta){ alpha = std::max(alpha, score); }

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

    return make_result(best_score, best_move);
  }


  auto root_search(position_history& hist, const board bd, const T& alpha, const T& beta, const search::depth_type depth) -> pv_search_result_t<T, true> const {
    assert(alpha < beta);
    return pv_search<true, true>(hist, evaluator_, bd, alpha, beta, depth);
  }


  void iterative_deepening_loop_(){
    for(;;){
      std::unique_lock<std::mutex> go_lk(go_mutex_);
      cv_.wait(go_lk, [this]{ return go_.load(std::memory_order_relaxed); });
      
      // store new search data locally 
      std::unique_lock<std::mutex> position_lk(position_mutex_);
      const auto bd = position_;
      auto hist = history_;
      position_lk.unlock();
      
      // iterative deepening
      auto alpha = -big_number<T>;
      auto beta = big_number<T>;
      for(; go_.load(std::memory_order_relaxed) && depth_.load() < (constants_ -> max_depth()); ++depth_){
        // increment table generation on every new root search
        tt_ -> update_gen();
      
        // update aspiration window once reasonable evaluation is obtained
        if(depth_.load(std::memory_order_relaxed) >= constants_ -> aspiration_depth()){
          const T previous_score = score();
          alpha = previous_score - aspiration_delta<T>;
          beta = previous_score + aspiration_delta<T>;
        }
        
        auto delta = aspiration_delta<T>;
        
        search::depth_type failed_high_count{0};
        for(;;){
          const search::depth_type adjusted_depth = std::max(1, depth_.load() - failed_high_count);
          const auto len_before = hist.history_.size();
          const auto[search_score, mv] = root_search(hist, bd, alpha, beta, adjusted_depth);
          const auto len_after = hist.history_.size();
          assert(len_before == len_after);
          
          if(!go_.load()){ break; }
          
          // update aspiration window if failing low or high
          if(search_score <= alpha){
            beta = (alpha + beta) / static_cast<T>(2);
            alpha = search_score - delta;
            failed_high_count = 0;
          }else if(search_score >= beta){
            beta = search_score + delta;
            ++failed_high_count;
          }else{
            //store updated information
            std::uint32_t as_uint32; std::memcpy(&as_uint32, &search_score, score_num_bytes);
            score_.store(as_uint32);
            best_move_.store(mv.data);
            break;
          }
          
          // exponentially grow window
          delta += delta / static_cast<T>(3);
        }
      }
      
      // stop search if we reach max_depth, otherwise, go_ is already false 
      go_.store(false);
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

  void go(const search::depth_type start_depth){
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

  thread_worker(
    const nnue::half_kp_weights<T>* weights,
    std::shared_ptr<table> tt,
    std::shared_ptr<sided_history_heuristic> hh,
    std::shared_ptr<search::constants> constants
  ) : 
    evaluator_(weights), tt_{tt}, hh_{hh}, constants_{constants}
  {
    std::thread([this]{ iterative_deepening_loop_(); }).detach();
  }
};

template<typename T>
struct worker_pool{
  
  const nnue::half_kp_weights<T>* weights_;
  std::shared_ptr<table> tt_{nullptr};
  std::shared_ptr<sided_history_heuristic> hh_{nullptr};
  std::shared_ptr<search::constants> constants_{nullptr};

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
    constants_ -> update_(new_size);
    const size_t new_workers = new_size - pool_.size();
    for(size_t i(0); i < new_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights_, tt_, hh_, constants_));
    }
  }

  void go(){
    for(size_t i(0); i < pool_.size(); ++i){
      const search::depth_type start_depth = 1 + static_cast<search::depth_type>(i % 2);
      pool_[i] -> go(start_depth);
    }
  }
  
  void stop(){
    for(auto& worker : pool_){ worker -> stop(); }
  }

  size_t nodes() const {
    return std::accumulate(pool_.cbegin(), pool_.cend(), static_cast<size_t>(0), [](const size_t& count, const auto& worker){
      return count + worker -> nodes();
    });
  }

  void set_position(const position_history& hist, const board& bd){
    (hh_ -> white).clear();
    (hh_ -> black).clear();
    for(auto& worker : pool_){ worker -> set_position(hist, bd); }
  }

  worker_pool(const nnue::half_kp_weights<T>* weights, size_t hash_table_size, size_t num_workers) : weights_{weights} {
    tt_ = std::make_shared<table>(hash_table_size);
    hh_ = std::make_shared<sided_history_heuristic>();
    constants_ = std::make_shared<search::constants>(num_workers);
    for(size_t i(0); i < num_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_, hh_, constants_));
    }
  }
};

}
