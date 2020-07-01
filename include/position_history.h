#pragma once

#include <vector>

#include <zobrist_util.h>

namespace chess{

struct position_history{
  std::vector<zobrist::hash_type> history_;

  position_history& clear(){
    history_.clear();
    return *this;
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

}
