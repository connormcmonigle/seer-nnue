#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>
#include <tuple>

#include <move.h>
#include <history_heuristic.h>

namespace chess{

struct move_orderer_iterator{
  using difference_type = long;
  using value_type = std::tuple<int, move>;
  using pointer = std::tuple<int, move>*;
  using reference = std::tuple<int, move>&;
  using iterator_category = std::output_iterator_tag;

  move_list list_{};
  const history_heuristic* hh_{nullptr};
  move first_{move::null()};

  int idx_{0};

  void update_list_(){
    auto best = [this](const size_t i0, const size_t i1){
      const move a = (list_.data)[i0];
      const move b = (list_.data)[i1];
      if(a.is_capture() && !b.is_capture()){
        return i0;
      }else if(!a.is_capture() && b.is_capture()){
        return i1;
      }else if(a.is_capture() && b.is_capture()){
        const int victim_a = static_cast<int>(a.captured());
        const int victim_b = static_cast<int>(b.captured());
        if(victim_a > victim_b){
          return i0;
        }else if(victim_a < victim_b){
          return i1;
        }else if(victim_a == victim_b){
          const int type_a = static_cast<int>(a.piece());
          const int type_b = static_cast<int>(b.piece());
          if(type_a < type_b){
            return i0;
          }else{
            return i1;
          }
        }
      }
      const auto a_count = hh_ -> count(a);
      const auto b_count = hh_ -> count(b);
      return a_count >= b_count ? i0 : i1;
    };

    size_t best_idx = idx_;
    for(size_t i(idx_); i < list_.size(); ++i){
      best_idx = best(best_idx, i);
    }
    std::swap(list_.data[best_idx], list_.data[idx_]);
  }
  
  move_orderer_iterator& operator++(){
    ++idx_;
    if(idx_ < static_cast<int>(list_.size())){ update_list_(); }
    return *this;
  }
  
  move_orderer_iterator operator++(int) {
    auto retval = *this;
    ++(*this);
    return retval;
  }
  
  bool operator==(const move_orderer_iterator& other) const {
    return other.idx_ == idx_;
  }
  
  bool operator!=(const move_orderer_iterator& other) const {
    return !(*this == other);
  }
  
  std::tuple<int, move> operator*() const {
    return std::tuple(idx_, (list_.data)[idx_]);
  }

  move_orderer_iterator(const move_list& list, const history_heuristic* hh, const move& first, const int& idx) : 
    list_{list}, hh_{hh}, first_{first}, idx_{idx}
  {
    if(!first.is_null()){
      const auto iter = std::find(list_.begin(), list_.end(), first_);
      if(iter != list_.end()){
        std::iter_swap(list_.begin(), iter);
      }else{
        update_list_();
      }
    }else{
      update_list_();
    }
  }
  
  move_orderer_iterator(const int& idx) : idx_{idx} {}
};

struct move_orderer{
  using iterator = move_orderer_iterator;
  
  move_list list_;
  const history_heuristic* hh_;

  move first{move::null()};

  move_orderer_iterator begin(){ return move_orderer_iterator(list_, hh_, first, 0); }
  move_orderer_iterator end(){ return move_orderer_iterator(list_.size()); }

  move_orderer& set_first(const move& mv){
    first = mv;
    return *this;
  }

  move_orderer(const move_list& list, const history_heuristic* hh) : list_{list}, hh_{hh} {}

};

}
