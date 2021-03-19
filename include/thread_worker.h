#pragma once

#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <cstring>
#include <vector>
#include <functional>

#include <nnue_model.h>
#include <board.h>
#include <move.h>
#include <search_constants.h>
#include <search_stack.h>
#include <move_orderer.h>
#include <transposition_table.h>
#include <history_heuristic.h>


namespace chess{


template<bool is_root> struct pv_search_result{};
template<> struct pv_search_result<false>{ using type = search::score_type; };
template<> struct pv_search_result<true>{ using type = std::tuple<search::score_type, move>; };

template<bool is_root>
using pv_search_result_t = typename pv_search_result<is_root>::type;

template<typename T, bool is_active=true>
struct thread_worker{

  std::mutex go_mutex_{};
  std::condition_variable cv_{};
  std::atomic<bool> go_{false};
  std::atomic<bool> is_stable_{false};
  std::atomic<search::depth_type> depth_{};
  std::atomic<size_t> nodes_{};
  
  std::atomic<search::score_type> score_{};
  std::atomic<std::uint32_t> best_move_{};

  sided_history_heuristic hh_{};
  
  nnue::eval<T> evaluator_;
  std::shared_ptr<transposition_table> tt_;
  std::shared_ptr<search::constants> constants_;
  std::function<void(const thread_worker<T, is_active>&)> iteration_callback;
  
  std::mutex stack_mutex_{};
  search::stack stack_;

  template<bool is_pv>
  search::score_type q_search(const search::stack_view& ss, const nnue::eval<T>& eval, const board& bd, search::score_type alpha, const search::score_type& beta, const search::depth_type& elevation){
    ++nodes_;
    const auto list = bd.generate_loud_moves();
    const bool is_check = bd.is_check();

    if(list.size() == 0 && is_check){ return ss.effective_mate_score(); }
    if(ss.is_two_fold(bd.hash())){ return search::draw_score; }
    if(bd.is_trivially_drawn()){ return search::draw_score; }

    auto orderer = move_orderer(move_orderer_data{move::null(), move::null(), move::null(), &bd, list, &hh_.us(bd.turn())});
    
    if(const std::optional<transposition_table_entry> maybe = tt_ -> find(bd.hash()); maybe.has_value()){
      const transposition_table_entry entry = maybe.value();
      const bool is_cutoff = 
        (entry.score() >= beta && entry.bound() == bound_type::lower) ||
        (entry.score() <= alpha && entry.bound() == bound_type::upper);
      if(is_cutoff){ return entry.score(); }
      orderer.set_first(entry.best_move());
    }

    const search::score_type static_eval = [&]{
      const search::score_type val = is_check ? ss.effective_mate_score() : eval.evaluate(bd.turn());
      if(maybe.has_value()){
        if(maybe -> bound() == bound_type::upper && val > maybe -> score()){ return maybe -> score(); }
        if(maybe -> bound() == bound_type::lower && val < maybe -> score()){ return maybe -> score(); }
      }
      return val;
    }();

    if(list.size() == 0 || static_eval >= beta){ return static_eval; }
    if(ss.reached_max_height()){ return static_eval; }
    
    alpha = std::max(alpha, static_eval);
    search::score_type best_score = static_eval;
    move best_move = move::null();
    
    ss.set_hash(bd.hash()).set_eval(static_eval);
    for(const auto& [idx, mv] : orderer){
      assert((mv != move::null()));
      if(!go_.load(std::memory_order_relaxed) || best_score >= beta){ break; }
      if(is_check || bd.see<search::see_type>(mv) >= 0){
        ss.set_played(mv);
        const nnue::eval<T> eval_ = bd.apply_update(mv, eval);
        const board bd_ = bd.forward(mv);
        
        const search::score_type score = -q_search<is_pv>(ss.next(), eval_, bd_, -beta, -alpha, elevation + 1);
        
        if(score > best_score){
          best_score = score;
          best_move = mv;
          if(score > alpha){
            if(score < beta){ alpha = score; }
            if constexpr(is_pv) { ss.prepend_to_pv(mv); }
          }
        }
      }
    }

    if(go_.load(std::memory_order_relaxed)){
      const auto bound = best_score >= beta ? bound_type::lower : bound_type::upper;
      const transposition_table_entry entry(bd.hash(), bound, best_score, best_move, 0);
      tt_-> insert(entry);
    }
    
    return best_score;
  }

  template<bool is_pv, bool is_root=false>
  auto pv_search(const search::stack_view& ss, const nnue::eval<T>& eval, const board& bd, search::score_type alpha, const search::score_type& beta, search::depth_type depth) -> pv_search_result_t<is_root> {
    auto make_result = [](const search::score_type& score, const move& mv){
      if constexpr(is_root){ return pv_search_result_t<is_root>{score, mv}; }
      if constexpr(!is_root){ return score; }
    };
    
    assert(depth >= 0);

    // step 1. drop into qsearch if depth reaches zero
    if(depth <= 0) { return make_result(q_search<is_pv>(ss, eval, bd, alpha, beta, 0), move::null()); }
    ++nodes_;

    // step 2. check if node is terminal
    const auto list = bd.generate_moves();
    const bool is_check = bd.is_check();
    if(list.size() == 0 && is_check){ return make_result(ss.effective_mate_score(), move::null()); }
    if(list.size() == 0) { return make_result(search::draw_score, move::null()); }
    if(!is_root && ss.is_two_fold(bd.hash())){ return make_result(search::draw_score, move::null()); }
    if(!is_root && bd.is_trivially_drawn()){ return make_result(search::draw_score, move::null()); }
  
    // step 3. initialize move orderer (setting tt move first if applicable)
    // and check for tt entry + tt induced cutoff on nonpv nodes
    const move killer = ss.killer();
    const move follow = ss.follow();
    const move counter = ss.counter();
    
    auto orderer = move_orderer(move_orderer_data{killer, follow, counter, &bd, list, &hh_.us(bd.turn())});
    const std::optional<transposition_table_entry> maybe = tt_ -> find(bd.hash());
    if(maybe.has_value()){
      const transposition_table_entry entry = maybe.value();
      const bool is_cutoff = !is_pv &&
        entry.depth() >= depth &&
        (entry.score() >= beta ? (entry.bound() == bound_type::lower) : (entry.bound() == bound_type::upper));
      if(is_cutoff){ return make_result(entry.score(), entry.best_move()); }
      orderer.set_first(entry.best_move());
    }
    
    // step 4. compute static eval and adjust appropriately if there's a tt hit
    const search::score_type static_eval = [&]{
      const search::score_type val = eval.evaluate(bd.turn());
      if(maybe.has_value()){
        if(maybe -> bound() == bound_type::upper && val > maybe -> score()){ return maybe -> score(); }
        if(maybe -> bound() == bound_type::lower && val < maybe -> score()){ return maybe -> score(); }
      }
      return val;
    }();

    // step 5. return static eval if max depth was reached
    if(ss.reached_max_height()){ return make_result(static_eval, move::null()); }
    
    // step 6. add position and static eval to stack
    ss.set_hash(bd.hash()).set_eval(static_eval);
    const bool improving = ss.improving();

    // step 7. static null move pruning
    const bool snm_prune = 
      !is_root && !is_pv && 
      !is_check &&
      depth <= constants_ -> snmp_depth() &&
      static_eval > beta + constants_ -> snmp_margin(improving, depth) &&
      static_eval > ss.effective_mate_score();

    if(snm_prune){ return make_result(static_eval, move::null()); }

    // step 8. null move pruning
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
      const search::score_type nmp_score = -pv_search<is_pv>(ss.next(), eval, bd.forward(move::null()), -beta, -alpha, adjusted_depth);
      if(nmp_score > beta){ return make_result(nmp_score, move::null()); }
    }
    
    // list of attempted quiets for updating histories
    move_list quiets_tried{};
    
    // move loop
    search::score_type best_score = ss.effective_mate_score();
    move best_move = *list.begin();

    for(const auto& [idx, mv] : orderer){
      assert((mv != move::null()));
      if(!go_.load(std::memory_order_relaxed) || best_score >= beta){ break; }
      ss.set_played(mv);
      
      const search::counter_type history_value = hh_.us(bd.turn()).compute_value(follow, counter, mv);
      
      const board bd_ = bd.forward(mv);
      
      const bool try_pruning = 
        !is_root && !is_pv && 
        !bd_.is_check() && !is_check &&
        idx != 0 && mv.is_quiet() &&
        best_score > ss.effective_mate_score();
      
      // step 9. pruning
      if(try_pruning){
        const bool history_prune = 
          depth <= constants_ -> history_prune_depth() &&
          history_value <= constants_ -> history_prune_threshold(improving, depth);
        
        if(history_prune){ continue; }
        
        const bool futility_prune = 
          depth <= constants_ -> futility_prune_depth() &&
          static_eval + constants_ -> futility_margin(depth) < alpha;
        
        if(futility_prune){ continue; }
      }

      const nnue::eval<T> eval_ = bd.apply_update(mv, eval);

      // step 10. extensions
      const search::depth_type extension = [&, mv = mv]{
        const bool check_ext = bd.see<search::see_type>(mv) > 0 && bd_.is_check();

        if(check_ext){ return 1; }
        
        const bool history_ext = 
          !is_root && maybe.has_value() && 
          mv == maybe -> best_move() && mv.is_quiet() &&
          depth >= constants_ -> history_extension_depth() &&
          history_value >= constants_ -> history_extension_threshold();
        
        if(history_ext){ return 1; }

        return 0;
      }();
      
      const search::score_type score = [&, this, idx = idx, mv = mv]{
        const search::depth_type next_depth = depth + extension - 1;
        auto full_width = [&]{ return -pv_search<is_pv>(ss.next(), eval_, bd_, -beta, -alpha, next_depth); };
          
        const bool try_lmr = 
          !is_check &&
          (mv.is_quiet() || bd.see<search::see_type>(mv) < 0) &&
          idx != 0 &&
          (depth >= constants_ -> reduce_depth());
        search::score_type zw_score{};
        
        // step 11. late move reductions
        if(try_lmr){
          search::depth_type reduction = constants_ -> reduction(depth, idx);
          
          // adjust reduction
          if(bd_.is_check()){ --reduction; }
          if(bd.is_passed_push(mv)){ --reduction; }
          if(!improving){ ++reduction; }
          if(!is_pv){ ++reduction; }
          if(bd.see<search::see_type>(mv) < 0 && mv.is_quiet()){ ++reduction; }

          if(mv.is_quiet()){ reduction += constants_ -> history_reduction(history_value); }
          
          reduction = std::max(reduction, 0);
          
          const search::depth_type lmr_depth = std::max(1, next_depth - reduction);
          zw_score = -pv_search<false>(ss.next(), eval_, bd_, -alpha - 1, -alpha, lmr_depth);
        }
        
        // search again at full depth if necessary
        if(!try_lmr || (try_lmr && (zw_score > alpha))){
          zw_score = -pv_search<false>(ss.next(), eval_, bd_, -alpha - 1, -alpha, next_depth);
        }
        
        // search again with full window on pv nodes
        const bool interior = zw_score > alpha && zw_score < beta;
        return (interior && is_pv) ? full_width() : zw_score;
      }();

      if(score < beta && mv.is_quiet()){ quiets_tried.add_(mv); }

      if(score > best_score){
        best_score = score;
        best_move = mv;
        if(score > alpha){
          if(score < beta){ alpha = score; }
          if constexpr(is_pv) { ss.prepend_to_pv(mv); }
        }
      }
    }
    
    // step 12. update histories if appropriate and maybe insert a new transposition_table_entry
    if(go_.load(std::memory_order_relaxed)){
      if(best_score >= beta){
        const transposition_table_entry entry(bd.hash(), bound_type::lower, best_score, best_move, depth);
        tt_-> insert(entry);
        if(best_move.is_quiet()){
          hh_.us(bd.turn()).update(follow, counter, best_move, quiets_tried, depth);
          ss.set_killer(best_move);
        }
      }else{
        const transposition_table_entry entry(bd.hash(), bound_type::upper, best_score, best_move, depth);
        tt_-> insert(entry);
      }
    }

    return make_result(best_score, best_move);
  }

  void iterative_deepening_loop_(){
    // acquire stack
    std::lock_guard<std::mutex> stack_lk(stack_mutex_);
      
    // iterative deepening
    search::score_type alpha = -search::big_number;
    search::score_type beta = search::big_number;
    for(; go_.load(std::memory_order_relaxed) && depth_.load() < (constants_ -> max_depth()); ++depth_){
      // update aspiration window once reasonable evaluation is obtained
      if(depth_.load(std::memory_order_relaxed) >= constants_ -> aspiration_depth()){
        const search::score_type previous_score = score();
        alpha = previous_score - search::aspiration_delta;
        beta = previous_score + search::aspiration_delta;
      }

      search::score_type delta = search::aspiration_delta;
      search::depth_type failed_high_count{0};

      for(;;){
        stack_.clear_future();

        const search::depth_type adjusted_depth = std::max(1, depth_.load() - failed_high_count);
        const auto [search_score, search_move] = pv_search<true, true>(search::stack_view::root(stack_), evaluator_, stack_.root_pos(), alpha, beta, adjusted_depth);
          
        if(!go_.load()){ break; }
          
        // update aspiration window if failing low or high
        if(search_score <= alpha){
          beta = (alpha + beta) / 2;
          alpha = search_score - delta;
          failed_high_count = 0;
        }else if(search_score >= beta){
          beta = search_score + delta;
          ++failed_high_count;
        }else{
          //store updated information
            
          is_stable_.store(
            std::abs(score() - search_score) <= search::stability_threshold && 
            best_move_.load() == search_move.data
          );
            
          score_.store(search_score);
          best_move_.store(search_move.data);
          break;
        }
          
          // exponentially grow window
          delta += delta / 3;
        }
        
        //callback on iteration completion
        if(go_.load(std::memory_order_relaxed)){ iteration_callback(*this); }
      }

      // stops search if we reached max_depth, otherwise, go_ is already false 
      go_.store(false);
  }
  
  void active_loop_(){
    for(;;){
      std::unique_lock<std::mutex> go_lk(go_mutex_);
      cv_.wait(go_lk, [this]{ return go_.load(std::memory_order_relaxed); });
      iterative_deepening_loop_();
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

  move best_move() const {
    return move{best_move_.load()};
  }

  search::score_type score() const {
    return score_.load();
  }

  void go(const search::depth_type& start_depth){
    std::unique_lock<std::mutex> go_lk(go_mutex_);
    
    depth_.store(start_depth);
    nodes_.store(0);
    go_.store(true);
    is_stable_.store(false);

    go_lk.unlock();
    cv_.notify_one();
  }

  void stop(){
    go_.store(false, std::memory_order_relaxed);
  }

  thread_worker<T, is_active>& set_position(const position_history& hist, const board& bd){
    std::lock_guard<std::mutex> stack_lk(stack_mutex_);
    bd.show_init(evaluator_);
    hh_.clear();
    stack_ = search::stack(hist, bd);
    return *this;
  }

  thread_worker(
    const nnue::weights<T>* weights,
    std::shared_ptr<transposition_table> tt,
    std::shared_ptr<search::constants> constants,
    std::function<void(const thread_worker<T, is_active>&)> callback = [](auto&&...){}
  ) : 
    evaluator_(weights), tt_{tt}, 
    constants_{constants}, iteration_callback{callback}, 
    stack_(position_history{}, board::start_pos())
  {
    if constexpr(is_active){ std::thread([this]{ active_loop_(); }).detach(); }
  }
};

template<typename T>
struct worker_pool{
  static constexpr size_t primary_id = 0;
  
  const nnue::weights<T>* weights_;
  std::shared_ptr<transposition_table> tt_{nullptr};
  std::shared_ptr<search::constants> constants_{nullptr};

  std::vector<std::shared_ptr<thread_worker<T>>> pool_{};

  void grow(size_t new_size){
    assert((new_size >= pool_.size()));
    constants_ -> update_(new_size);
    const size_t new_workers = new_size - pool_.size();
    for(size_t i(0); i < new_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights_, tt_, constants_));
    }
  }

  void go(){
    // increment table generation at start of search
    tt_ -> update_gen();
    
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
    return *pool_[primary_id];
  }

  worker_pool(
    const nnue::weights<T>* weights, 
    size_t hash_table_size, size_t num_workers, 
    std::function<void(const thread_worker<T, true>&)> primary_callback = [](auto&&...){}
  ) : 
      weights_{weights} 
  {
    tt_ = std::make_shared<transposition_table>(hash_table_size);
    constants_ = std::make_shared<search::constants>(num_workers);
    
    pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_, constants_, primary_callback));
    for(size_t i(1); i < num_workers; ++i){
      pool_.push_back(std::make_shared<thread_worker<T>>(weights, tt_, constants_));
    }
  }
};

}
