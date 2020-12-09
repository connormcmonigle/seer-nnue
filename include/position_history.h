#pragma once

#include <vector>

#include <zobrist_util.h>
#include <move.h>

namespace chess{

template<typename T>
struct popper{
  T* data_;
  popper(T* data) : data_{data} {}
  ~popper(){ data_ -> pop_(); }
};

template<typename T, typename U>
struct base_history{
  using stored_type = U;
  std::vector<stored_type> history_;

  T& cast(){ return static_cast<T&>(*this); }
  const T& cast() const { return static_cast<const T&>(*this); }

  T& clear(){
    history_.clear();
    return cast();
  }

  popper<T> scoped_push_(const stored_type& elem){
    history_.push_back(elem);
    return popper<T>(&cast());
  }
  
  T& push_(const stored_type& elem){
    history_.push_back(elem);
    return cast();
  }
  
  T& pop_(){
    history_.pop_back();
    return cast();
  }
  
  stored_type back() const { return history_.back(); }

  size_t len() const {
    return history_.size();
  }

  base_history() : history_{} {}
  base_history(const std::vector<stored_type>& h) : history_{h} {}
};


struct position_history : base_history<position_history, zobrist::hash_type>{
  
  size_t occurrences(const zobrist::hash_type& hash) const {
    size_t occurrences_{0};
    for(auto it = history_.crbegin(); it != history_.crend(); ++it){
      occurrences_ += static_cast<size_t>(*it == hash);
    }
    return occurrences_;
  }

  bool is_three_fold(const zobrist::hash_type& hash) const {
    return occurrences(hash) >= 2;
  }
};


}
