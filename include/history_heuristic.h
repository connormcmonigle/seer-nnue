#pragma once

#include <iostream>
#include <array>
#include <type_traits>
#include <algorithm>

#include <enum_util.h>
#include <search_util.h>
#include <position_history.h>
#include <move.h>

namespace chess{

struct history_heuristic{
  using value_type = int;
  static constexpr size_t num_squares = 64;
  static constexpr size_t num_pieces = 6;

  std::array<value_type, num_squares * num_squares> butterfly_{};
  std::array<value_type, num_pieces * num_squares * num_pieces * num_squares> counter_{};
  std::array<value_type, num_pieces * num_squares * num_pieces * num_squares> follow_{};

  size_t butterfly_idx_(const move& mv) const {
    const size_t from = static_cast<size_t>(mv.from().index());
    const size_t to = static_cast<size_t>(mv.to().index());
    return from * num_squares + to;
  }

  size_t counter_idx_(const move& them_mv, const move& mv) const {
    const size_t p0 = static_cast<size_t>(them_mv.piece());
    const size_t to0 = static_cast<size_t>(them_mv.from().index());
    const size_t p1 = static_cast<size_t>(mv.piece());
    const size_t to1 = static_cast<size_t>(mv.to().index());
    return p0 * num_squares * num_pieces * num_squares + to0 * num_pieces * num_squares + p1 * num_squares + to1;
  }

  size_t follow_idx_(const move& us_mv, const move& mv) const {
    const size_t p0 = static_cast<size_t>(us_mv.piece());
    const size_t to0 = static_cast<size_t>(us_mv.from().index());
    const size_t p1 = static_cast<size_t>(mv.piece());
    const size_t to1 = static_cast<size_t>(mv.to().index());
    return p0 * num_squares * num_pieces * num_squares + to0 * num_pieces * num_squares + p1 * num_squares + to1;
  }

  history_heuristic& clear(){
    butterfly_.fill(value_type{});
    counter_.fill(value_type{});
    follow_.fill(value_type{});
    return *this;
  }

  history_heuristic& update(const move& follow, const move& counter, const move& best_move, const move_list& tried, const search::depth_type& depth){
    // more or less lifted from ethereal
    constexpr value_type history_max = 400;
    constexpr value_type history_multiplier = 32;
    constexpr value_type history_divisor = 512;
    assert((!tried.has(best_move)));
    auto single_update = [&, this](const auto& mv, const value_type& gain){
      // update butterfly history
      {
        const size_t idx = butterfly_idx_(mv);
        butterfly_[idx] += (gain * history_multiplier) - (butterfly_[idx] * std::abs(gain) / history_divisor);
      }
      // update counter move history
      if(!counter.is_null()){
        const size_t idx = counter_idx_(counter, mv);
        counter_[idx] += (gain * history_multiplier) - (counter_[idx] * std::abs(gain) / history_divisor);
      }
      // update follow up move history
      if(!follow.is_null()){
        const size_t idx = follow_idx_(follow, mv);
        follow_[idx] += (gain * history_multiplier) - (follow_[idx] * std::abs(gain) / history_divisor);
      }
    };
    // limit gain to prevent saturation
    const value_type gain = std::min(history_max, static_cast<value_type>(depth) * static_cast<value_type>(depth));
    std::for_each(tried.cbegin(), tried.cend(), [single_update, gain, this](const move& mv){ single_update(mv, -gain); });
    single_update(best_move, gain);
    return *this;
  }

  value_type compute_value(const move& follow, const move& counter, const move& mv) const {
    return 
      (follow.is_null() ? value_type{} : follow_[follow_idx_(follow, mv)]) + 
      (counter.is_null() ? value_type{} : counter_[counter_idx_(counter, mv)]) +
      butterfly_[butterfly_idx_(mv)];
  }

};

struct sided_history_heuristic : sided<sided_history_heuristic, history_heuristic> {
  history_heuristic white;
  history_heuristic black;
  
  sided_history_heuristic& clear(){
    white.clear();
    black.clear();
    return *this;
  }
  
  sided_history_heuristic() : white{}, black{} {}
};

std::ostream& operator<<(std::ostream& ostr, const history_heuristic& hh){
  ostr << "butterfly: {";
  for(const auto& val : hh.butterfly_){ ostr << val << ", "; }
  ostr << "}\n\n";
  ostr << "counter: {";
  for(const auto& val : hh.counter_){ ostr << val << ", "; }
  ostr << "}\n\n";
  ostr << "follow: {";
  for(const auto& val : hh.follow_){ ostr << val << ", "; }
  return ostr << "}\n\n";
}

}
