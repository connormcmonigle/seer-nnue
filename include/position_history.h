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

struct position_history{
  std::vector<zobrist::hash_type> history_;

  position_history& clear(){
    history_.clear();
    return *this;
  }

  popper<position_history> scoped_push_(const zobrist::hash_type& hash){
    history_.push_back(hash);
    return popper<position_history>(this);
  }
  
  position_history& push_(const zobrist::hash_type& hash){
    history_.push_back(hash);
    return *this;
  }
  
  position_history& pop_(){
    history_.pop_back();
    return *this;
  }
  
  zobrist::hash_type back() const { return history_.back(); }
  
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

  position_history() : history_{} {}
  position_history(std::vector<zobrist::hash_type>& h) : history_{h} {}
};


struct move_history{
  std::vector<chess::move> history_;

  move_history& clear(){
    history_.clear();
    return *this;
  }

  popper<move_history> scoped_push_(const chess::move& mv){
    history_.push_back(mv);
    return popper<move_history>(this);
  }
  
  move_history& push_(const chess::move& mv){
    history_.push_back(mv);
    return *this;
  }
  
  move_history& pop_(){
    history_.pop_back();
    return *this;
  }

  bool nmp_valid() const {
    return 
      (history_.size() >= 2) &&
      !(history_.rbegin() -> is_null()) &&
      !((history_.rbegin()+1) -> is_null());
  }
  
  move back() const { return history_.back(); }

  move_history() : history_{} {}
  move_history(std::vector<chess::move>& h) : history_{h} {}
};

}
