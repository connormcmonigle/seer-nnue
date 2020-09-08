#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>
#include <tuple>

#include <move.h>
#include <history_heuristic.h>

namespace chess{

struct move_orderer_data{
  move follow{};
  move counter{};
  const board* bd{nullptr};
  move_list list{};
  const history_heuristic* hh{nullptr};
  move first{move::null()};

  move_orderer_data(const move& follow_, const move& counter_, const board* bd_, const move_list& list_, const history_heuristic* hh_) : 
  follow{follow_}, counter{counter_}, bd{bd_}, list{list_}, hh{hh_} {}
  
  move_orderer_data(){}
};

struct move_orderer_iterator{
  using difference_type = long;
  using value_type = std::tuple<int, move>;
  using pointer = std::tuple<int, move>*;
  using reference = std::tuple<int, move>&;
  using iterator_category = std::output_iterator_tag;


  move_orderer_data data_;

  int idx_{0};

  void update_list_(){
    auto best = [this](const size_t i0, const size_t i1){
      const move a = data_.list.data[i0];
      const move b = data_.list.data[i1];
      if(!a.is_quiet() && b.is_quiet()){
        return i0;
      }else if(a.is_quiet() && !b.is_quiet()){
        return i1;
      }else if(!a.is_quiet() && !b.is_quiet()){
        const int a_score = data_.bd -> see<int>(a);
        const int b_score = data_.bd -> see<int>(b);
        return (a_score > b_score) ? i0 : i1;
      }
      const auto a_value = data_.hh -> compute_value(data_.follow, data_.counter, a);
      const auto b_value = data_.hh -> compute_value(data_.follow, data_.counter, b);
      return a_value >= b_value ? i0 : i1;
    };

    size_t best_idx = idx_;
    for(size_t i(idx_); i < data_.list.size(); ++i){
      best_idx = best(best_idx, i);
    }
    std::swap(data_.list.data[best_idx], data_.list.data[idx_]);
  }
  
  move_orderer_iterator& operator++(){
    ++idx_;
    if(idx_ < static_cast<int>(data_.list.size())){ update_list_(); }
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
    return std::tuple(idx_, (data_.list.data)[idx_]);
  }

  move_orderer_iterator(const move_orderer_data& data, const int& idx) : data_{data}, idx_{idx}
  {
    if(!data_.first.is_null()){
      const auto iter = std::find(data_.list.begin(), data_.list.end(), data_.first);
      if(iter != data_.list.end()){
        std::iter_swap(data_.list.begin(), iter);
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
  
  move_orderer_data data_;

  move_orderer_iterator begin(){ return move_orderer_iterator(data_, 0); }
  move_orderer_iterator end(){ return move_orderer_iterator(data_.list.size()); }

  move_orderer& set_first(const move& mv){
    data_.first = mv;
    return *this;
  }

  move_orderer(const move_orderer_data& data) : data_{data} {}

};

}
