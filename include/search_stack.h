#pragma once

#include <array>


#include <search_util.h>
#include <enum_util.h>
#include <zobrist_util.h>
#include <position_history.h>
#include <move.h>

namespace search{

template<typename T>
struct stack_entry{
  zobrist::hash_type hash_{};
  T eval_{};
  chess::move played_{chess::move::null()};
  chess::move killer_{chess::move::null()};
};

template<typename T>
struct stack{
  static constexpr depth_type margin = 16;
  chess::position_history past_;
  std::array<stack_entry<T>, max_depth_ + margin> present_{};
  
  stack_entry<T>& at_(const depth_type& height){
    return present_[height];
  }

  size_t occurrences(const size_t& height, const zobrist::hash_type& hash) const {
    size_t occurrences_{0};
    for(auto it = present_.cbegin(); it != (present_.cbegin() + height); ++it){
      occurrences_ += static_cast<size_t>(it -> hash_ == hash);
    }
    return occurrences_ + past_.occurrences(hash);
  }
  
  stack(const chess::position_history& hist) : past_{hist} {}
};

template<typename T>
struct stack_view{
  stack<T>* view_;
  depth_type height_{};

  bool is_three_fold(const zobrist::hash_type& hash) const {
    return view_ -> occurrences(height_, hash) >= 2;
  }

  chess::move counter() const {
    if(height_ <= 0){ return chess::move::null(); }
    return (view_ -> at_(height_ - 1)).played_;
  }
  
  chess::move follow() const {
    if(height_ <= 1){ return chess::move::null(); }
    return (view_ -> at_(height_ - 2)).played_;
  }
  
  chess::move killer() const {
    return (view_ -> at_(height_)).killer_;
  }
  
  bool nmp_valid() const {
    return !counter().is_null() && !follow().is_null();
  }
  
  bool improving() const {
    return (height_ > 1) && (view_ -> at_(height_ - 2)).eval_ < (view_ -> at_(height_)).eval_;
  }

  const stack_view<T>& set_hash(const zobrist::hash_type& hash) const {
    (view_ -> at_(height_)).hash_ = hash;
    return *this;
  }

  const stack_view<T>& set_eval(const T& eval) const {
    (view_ -> at_(height_)).eval_ = eval;
    return *this;
  }

  const stack_view<T>& set_played(const chess::move& played) const {
    (view_ -> at_(height_)).played_ = played;
    return *this;
  }

  const stack_view<T>& set_killer(const chess::move& killer) const {
    (view_ -> at_(height_)).killer_ = killer;
    return *this;
  }

  stack_view<T> prev() const { return stack_view<T>(view_, height_ - 1); }
  
  stack_view<T> next() const { return stack_view<T>(view_, height_ + 1); }
  
  stack_view(stack<T>* view, const depth_type& height) : 
    view_{view},
    height_{std::min(max_depth_ + stack<T>::margin - 1, height)} 
  {
    assert((height >= 0));
  }
  
  static stack_view root(stack<T>& st){ return stack_view(&st, 0); }
};


}

