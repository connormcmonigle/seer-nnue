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
#include <search_stack.h>
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
  std::atomic<bool> is_stable_{false};
  std::atomic<search::depth_type> depth_;
  std::atomic<size_t> nodes_;
  std::atomic<size_t> nodes_at_depth_;
  
  std::atomic<std::uint32_t> score_;
  std::atomic<std::uint32_t> best_move_{};

  nnue::half_kp_eval<T> evaluator_;
  sided_history_heuristic hh_;

  std::shared_ptr<table> tt_;
  std::shared_ptr<search::constants> constants_;

  std::mutex position_mutex_{};
  board position_{};
  position_history history_{};

  T q_search(const search::stack_view<T>& ss, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T& beta, const search::depth_type& elevation){
    ++nodes_;
    
    const auto all_list = bd.generate_moves();

    const bool is_check = bd.is_check();
    if(all_list.size() == 0 && is_check){ return mate_score<T>; }
    if(all_list.size() == 0) { return draw_score<T>; }
    if(ss.is_three_fold(bd.hash())){ return draw_score<T>; }
    
    const auto list = all_list.loud();
    auto orderer = move_orderer(move_orderer_data{move::null(), move::null(), move::null(), &bd, list, &hh_.us(bd.turn())});
    
    if(const std::optional<tt_entry> maybe = tt_ -> find(bd.hash()); maybe.has_value()){
      const tt_entry entry = maybe.value();
      const bool is_cutoff = 
        (entry.score() >= beta && entry.bound() == bound_type::lower) ||
        (entry.score() <= alpha && entry.bound() == bound_type::upper);
      if(is_cutoff){ return entry.score(); }
      alpha = std::max(alpha, entry.score());
      orderer.set_first(entry.best_move());
    }

    const T static_eval = eval.propagate(bd.turn());
    if(list.size() == 0 || static_eval > beta){ return static_eval; }

    alpha = std::max(alpha, static_eval);
    T best_score = static_eval;
    
    ss.set_hash(bd.hash()).set_eval(static_eval);

    for(auto [idx, mv] : orderer){
      assert((mv != move::null()));
      if(!go_.load(std::memory_order_relaxed) || best_score > beta){ break; }
      if(bd.see<int>(mv) >= 0){
        const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
        const board bd_ = bd.forward(mv);
      
        const T score = -q_search(ss.next(), eval_, bd_, -beta, -alpha, elevation + 1);
        alpha = std::max(alpha, score);
        best_score = std::max(best_score, score);
      }
    }

    return best_score;
  }

  template<bool is_pv, bool is_root=false>
  auto pv_search(const search::stack_view<T>& ss, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T& beta, search::depth_type depth) -> pv_search_result_t<T, is_root>{
    auto make_result = [](const T& score, const move& mv){
      if constexpr(is_root){ return pv_search_result_t<T, is_root>{score, mv}; }
      if constexpr(!is_root){ return score; }
    };
    
    assert(depth >= 0);
    
    // step 1. check if node is terminal
    const auto list = bd.generate_moves();
    const bool is_check = bd.is_check();
    if(list.size() == 0 && is_check){ return make_result(mate_score<T>, move::null()); }
    if(list.size() == 0) { return make_result(draw_score<T>, move::null()); }
    if(ss.is_three_fold(bd.hash())){ return make_result(draw_score<T>, move::null()); }
    
    // don't drop into qsearch if in check
    if(is_check && depth <= 0){ depth = 1; }
  
    // step 2. drop into qsearch if depth reaches zero
    if(depth <= 0) { return make_result(q_search(ss, eval, bd, alpha, beta,  0), move::null()); }
    ++nodes_;

    // step 3. initialize move orderer (setting tt move first if applicable)
    // and check for tt entry + tt induced cutoff on nonpv nodes
    const move killer = ss.killer();
    const move follow = ss.follow();
    const move counter = ss.counter();
    
    auto orderer = move_orderer(move_orderer_data{killer, follow, counter, &bd, list, &hh_.us(bd.turn())});
    const std::optional<tt_entry> maybe = tt_ -> find(bd.hash());
    if(maybe.has_value()){
      const tt_entry entry = maybe.value();
      const bool is_cutoff = !is_pv &&
        entry.depth() >= depth &&
        (entry.score() >= beta ?
          (entry.bound() == bound_type::lower) :
          (entry.bound() == bound_type::upper));
      if(is_cutoff){ return make_result(entry.score(), entry.best_move()); }
      orderer.set_first(entry.best_move());
    }
    
    // step 4. compute static eval and adjust appropriately if there's a tt hit
    const T static_eval = [&]{
      const T val = eval.propagate(bd.turn());
      if(maybe.has_value()){
        if(maybe -> bound() == bound_type::upper && val > maybe -> score()){ return maybe -> score(); }
        if(maybe -> bound() == bound_type::lower && val < maybe -> score()){ return maybe -> score(); }
      }
      return val;
    }();

    // step 5. add position and static eval to stack
    ss.set_hash(bd.hash()).set_eval(static_eval);
    const bool improving = ss.improving();

    // step 6. static null move pruning
    const bool snm_prune = 
      !is_root && !is_pv && 
      !is_check && 
      depth <= constants_ -> snmp_depth() &&
      static_eval > beta + constants_ -> snmp_margin<T>(improving, depth) &&
      static_eval > mate_score<T>;

    if(snm_prune){ return make_result(static_eval, move::null()); }

    // step 7. null move pruning
    const bool try_nmp = 
      !is_root && !is_pv && 
      !is_check && 
      depth >= constants_ -> nmp_depth() &&
      static_eval > beta &&
      ss.nmp_valid() &&
      bd.has_non_pawn_material();

    if(try_nmp){
      ss.set_played(move::null());
      const search::depth_type R = constants_ -> R(depth);
      const search::depth_type adjusted_depth = std::max(0, depth - R);
      const T nmp_score = -pv_search<is_pv>(ss.next(), eval, bd.forward(move::null()), -beta, -alpha, adjusted_depth);
      if(nmp_score > beta){ return make_result(nmp_score, move::null()); }
    }
    
    // list of attempted quiets for updating histories
    move_list quiets_tried{};
    
    // move loop
    T best_score = mate_score<T>;
    move best_move = list.data[0];

    for(auto [idx, mv] : orderer){
      assert((mv != move::null()));
      if(!go_.load(std::memory_order_relaxed) || best_score > beta){ break; }
      ss.set_played(mv);
      
      const history_heuristic::value_type history_value = hh_.us(bd.turn()).compute_value(follow, counter, mv);
      
      const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
      const board bd_ = bd.forward(mv);
      
      const bool try_pruning = 
        !is_root && !is_pv && 
        !bd_.is_check() && !is_check &&
        idx != 0 && mv.is_quiet() &&
        best_score > mate_score<T>;
      
      // step 8. pruning
      if(try_pruning){
        const bool history_prune = 
          depth <= constants_ -> history_prune_depth() &&
          history_value <= constants_ -> history_prune_threshold<history_heuristic::value_type>();
        
        if(history_prune){ continue; }
      }
      
      // step 9. extensions
      const search::depth_type extension = [&]{
        if(bd.see<int>(mv) > 0 && bd_.is_check()){ return 1; }
        return 0;
      }();
      
      const T score = [&, this]{
        const search::depth_type next_depth = depth + extension - 1;
        auto full_width = [&]{ return -pv_search<is_pv>(ss.next(), eval_, bd_, -beta, -alpha, next_depth); };
          
        const bool try_lmr = 
          !is_check &&
          (mv.is_quiet() || bd.see<int>(mv) < 0) &&
          idx != 0 &&
          (depth >= constants_ -> reduce_depth());
        T zw_score{};
        
        // step 10. late move reductions
        if(try_lmr){
          search::depth_type reduction = constants_ -> reduction<is_pv>(depth, idx);
          
          // adjust reduction
          if(bd_.is_check()){ --reduction; }
          if(bd.is_passed_push(mv)){ --reduction; }
          if(!improving){ ++reduction; }
          if(!is_pv){ ++reduction; }
          if(bd.see<int>(mv) < 0){ ++reduction; }

          reduction += constants_ -> history_reduction(history_value);
          
          reduction = std::max(reduction, 0);
          
          const search::depth_type lmr_depth = std::max(1, next_depth - reduction);
          zw_score = -pv_search<false>(ss.next(), eval_, bd_, -alpha - epsilon<T>, -alpha, lmr_depth);
        }
        
        // search again at full depth if necessary
        if(!try_lmr || (try_lmr && (zw_score > alpha))){
          zw_score = -pv_search<false>(ss.next(), eval_, bd_, -alpha - epsilon<T>, -alpha, next_depth);
        }
        
        // search again with full window on pv nodes
        const bool interior = zw_score > alpha && zw_score < beta;
        return (interior && is_pv) ? full_width() : zw_score;
      }();

      if(score < beta){
        if(mv.is_quiet()){ quiets_tried.add_(mv); }
        alpha = std::max(alpha, score);
      }

      if(score > best_score){
        best_score = score;
        best_move = mv;
      }
    }
    
    // step 11. update histories if appropriate and maybe insert a new tt_entry
    if(go_.load(std::memory_order_relaxed)){
      if(best_score > beta){
        const tt_entry entry(bd.hash(), bound_type::lower, best_score, best_move, depth);
        tt_-> insert(entry);
        if(best_move.is_quiet()){
          hh_.us(bd.turn()).update(follow, counter, best_move, quiets_tried, depth);
          ss.set_killer(best_move);
        }
      }else{
        const tt_entry entry(bd.hash(), bound_type::upper, best_score, best_move, depth);
        tt_-> insert(entry);
      }
    }

    return make_result(best_score, best_move);
  }


  auto root_search(const position_history& pos_hist, const board bd, const T& alpha, const T& beta, const search::depth_type depth) -> pv_search_result_t<T, true> const {
    assert(alpha < beta);
    search::stack<T> record(pos_hist);
    auto result = pv_search<true, true>(search::stack_view<T>::root(record), evaluator_, bd, alpha, beta, depth);
    return result;
  }


  void iterative_deepening_loop_(){
    for(;;){
      std::unique_lock<std::mutex> go_lk(go_mutex_);
      cv_.wait(go_lk, [this]{ return go_.load(std::memory_order_relaxed); });
      
      // store new search data locally 
      std::unique_lock<std::mutex> position_lk(position_mutex_);
      const auto bd = position_;
      const auto hist = history_;
      position_lk.unlock();
      
      // iterative deepening
      auto alpha = -big_number<T>;
      auto beta = big_number<T>;
      for(; go_.load(std::memory_order_relaxed) && depth_.load() < (constants_ -> max_depth()); ++depth_){
        // update nodes_at_depth_
        nodes_at_depth_.store(nodes_.load());
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
            constexpr T stability_threshold = static_cast<T>(0.05);
            
            is_stable_.store(
              std::abs(score() - search_score) <= stability_threshold && 
              best_move_.load() == mv.data
            );
            
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

  bool is_stable() const {
    return is_stable_.load();
  }

  int depth() const {
    return depth_.load();
  }

  size_t nodes() const {
    return nodes_.load();
  }

  size_t nodes_at_depth() const {
    return nodes_at_depth_.load();
  }

  move best_move() const {
    return move{best_move_.load()};
  }

  T score() const {
    const std::uint32_t raw = score_.load();
    T result; std::memcpy(&result, &raw, score_num_bytes);
    return result;
  }

  void go(const search::depth_type& start_depth){
    std::unique_lock<std::mutex> go_lk(go_mutex_);
    depth_.store(start_depth);
    nodes_.store(0);
    nodes_at_depth_.store(0);
    go_.store(true);
    is_stable_.store(false);
    go_lk.unlock();
    cv_.notify_one();
  }

  void stop(){
    go_.store(false, std::memory_order_relaxed);
  }


  thread_worker<T>& set_position(const position_history& hist, const board& bd){
    std::lock_guard<std::mutex> position_lk(position_mutex_);
    bd.show_init(evaluator_);
    hh_.clear();
    history_ = hist;
    position_ = bd;
    return *this;
  }

  thread_worker(
    const nnue::half_kp_weights<T>* weights,
    std::shared_ptr<table> tt,
    std::shared_ptr<search::constants> constants
  ) : 
    evaluator_(weights), hh_{}, tt_{tt}, constants_{constants}
  {
    std::thread([this]{ iterative_deepening_loop_(); }).detach();
  }
};

template<typename T>
struct worker_pool{
  
  const nnue::half_kp_weights<T>* weights_;
  std::shared_ptr<table> tt_{nullptr};
  std::shared_ptr<search::constants> constants_{nullptr};

  std::vector<std::shared_ptr<thread_worker<T>>> pool_{};

  std::string pv_string(chess::board bd) const {
    std::string result{};
    constexpr size_t max_pv_length = 256;
    for(size_t i(0); i < max_pv_length; ++i){
      const std::optional<tt_entry> entry = tt_ -> find(bd.hash());
      if(!entry.has_value() || !bd.generate_moves().has(entry -> best_move())){ break; }
      result += entry -> best_move().name(bd.turn()) + " ";
      bd = bd.forward(entry -> best_move());
    }
    return result;
  }

  void grow(size_t new_size){
    assert((new_size >= pool_.size()));
    constants_ -> update_(new_size);
    const size_t new_workers = new_size - pool_.size();
    for(size_t i(0); i < new_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights_, tt_, constants_));
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
    for(auto& worker : pool_){ worker -> set_position(hist, bd); }
  }

  thread_worker<T>& primary_worker(){
    return *pool_[0];
  }

  worker_pool(const nnue::half_kp_weights<T>* weights, size_t hash_table_size, size_t num_workers) : weights_{weights} {
    tt_ = std::make_shared<table>(hash_table_size);
    constants_ = std::make_shared<search::constants>(num_workers);
    for(size_t i(0); i < num_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_, constants_));
    }
  }
};

}
