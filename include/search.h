#pragma once

#include <memory>
#include <tuple>
#include <limits>
#include <algorithm>

#include <nnue_half_kp.h>
#include <move.h>
#include <transposition_table.h>
#include <board.h>


namespace chess{

template<typename T>
inline constexpr T eta = static_cast<T>(0.005);

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

template<typename T, bool is_pv, bool is_root=false>
auto pv_search(std::shared_ptr<table> tt, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T beta, const int depth) -> pvs_result_t<T, is_root> {
  auto make_result = [](const T& score, const move& mv){
    if constexpr(is_root){
      return pvs_result_t<T, is_root>{score, mv};
    }else{
      return score;
    }
  };

  const auto list = bd.generate_moves();
  const auto empty_move = move{};

  if(list.size() == 0 && bd.is_check()){ return make_result(mate_score<T>, empty_move); }
  if(list.size() == 0) { return make_result(draw_score<T>, empty_move); }
  if(depth <= 0) { return make_result(eval.propagate(bd.turn()), empty_move); }

  T best_score = mate_score<T>;
  move best_move = *list.begin();

  if(const auto it = tt -> find(bd.hash()); it != tt -> end()){
    const tt_entry entry = *it;
    if(!is_pv && entry.depth() >= depth){
      if(entry.score() >= beta ? (entry.bound() == bound_type::lower) : (entry.bound() == bound_type::upper)){
        return make_result(entry.score(), entry.best_move());
      }
    }else if(list.has(entry.best_move())){
      best_move = entry.best_move();
    }
  }

  {
    const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(best_move, eval);
    const board bd_ = bd.forward(best_move);
    best_score = -pv_search<T, true>(tt, eval_, bd_, -beta, -alpha, depth - 1);
    alpha = std::max(alpha, best_score);
  }

  for(const move& mv : list){
    
    if(best_score > beta){ break; }
    if(mv == best_move){ continue; }

    const nnue::half_kp_eval<T> eval_ = bd.half_kp_updated(mv, eval);
    const board bd_ = bd.forward(mv);
    T score = -pv_search<T, false>(tt, eval_, bd_, -alpha - eta<T>, -alpha, depth - 1);

    if(score > alpha && score < beta){
      score = -pv_search<T, true>(tt, eval_, bd_, -beta, -alpha, depth - 1);
      alpha = std::max(alpha, score);
    }

    if(score > best_score){
      best_score = score;
      best_move = mv;
    }
  }

  if(best_score > beta){
    const tt_entry entry(bd.hash(), bound_type::lower, best_score, best_move, depth);
    tt -> insert(entry);
  }else{
    const tt_entry entry(bd.hash(), bound_type::upper, best_score, best_move, depth);
    tt -> insert(entry);
  }

  return make_result(best_score, best_move);
}

template<typename T>
auto pv_search(std::shared_ptr<table> tt, const nnue::half_kp_eval<T>& eval, const board& bd, const int depth) -> pvs_result_t<T, true> {
  constexpr T alpha = -big_number<T>;
  constexpr T beta = big_number<T>;
  return pv_search<T, true, true>(tt, eval, bd, alpha, beta, depth);
}

}