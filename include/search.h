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
T pv_search_impl(std::shared_ptr<table> tt, const nnue::half_kp_eval<T>& eval, const board& bd, T alpha, const T beta, const int depth){
  if(bd.is_check_mate()){ return mate_score<T>; }
  if(depth <= 0) { return eval.propagate(bd.turn()); }

  if(auto it = tt -> find(bd.hash()); it != tt -> end()){
    const tt_entry entry = *it;

    const auto eval_ = bd.half_kp_updated(entry.best_move(), eval);
    const auto bd_ = bd.forward(entry.best_move());
    const T score = -pv_search_impl(tt, eval_, bd_, -beta, -alpha, depth - 1);

    if(score > beta){ return beta; }
    alpha = std::max(alpha, score);
  }

  for(const move& mv : bd.generate_moves()){
    const auto eval_ = bd.half_kp_updated(mv, eval);
    const auto bd_ = bd.forward(mv);
    const T null_window = -pv_search_impl(tt, eval_, bd_, -alpha - eta<T>, -alpha, depth - 1);
    if(null_window > alpha){
      const T full_window = -pv_search_impl(tt, eval_, bd_, -beta, -alpha, depth - 1);
      tt -> insert(tt_entry{bd_.hash(), mv, depth});
      if(full_window > beta){ return beta; }
      alpha = full_window;
    }
  }

  return alpha;
}

template<typename T>
std::tuple<T, move> pv_search(std::shared_ptr<table> tt, const nnue::half_kp_eval<T>& eval, const board& bd, const int depth){
  const T beta = big_number<T>;
  T alpha = -beta;

  const auto move_list = bd.generate_moves();
  move best_move = move_list.data[0];

  if(auto it = tt -> find(bd.hash()); it != tt -> end()){
    const tt_entry entry = *it;

    const auto eval_ = bd.half_kp_updated(entry.best_move(), eval);
    const auto bd_ = bd.forward(entry.best_move());
    const T score = -pv_search_impl(tt, eval_, bd_, -beta, -alpha, depth);

    alpha = score;
    best_move = entry.best_move();
  }

  for(const move& mv : move_list){
    const auto eval_ = bd.half_kp_updated(mv, eval);
    const auto bd_ = bd.forward(mv);
    const T null_window = -pv_search_impl(tt, eval_, bd_, -alpha - eta<T>, -alpha, depth);
    if(null_window > alpha){
      const T full_window = -pv_search_impl(tt, eval_, bd_, -beta, -alpha, depth);
      tt -> insert(tt_entry{bd_.hash(), mv, depth});
      alpha = std::min(beta, full_window);
      best_move = mv;
    }
  }

  return std::tuple(alpha, best_move);
}

}