#pragma once

#include <array>
#include <string>
#include <algorithm>

#include <search_util.h>
#include <enum_util.h>
#include <zobrist_util.h>
#include <position_history.h>
#include <move.h>
#include <board.h>

namespace search{

constexpr depth_type safe_depth_ = max_depth_ + max_depth_margin_;

struct stack_entry{
  zobrist::hash_type hash_{};
  search::score_type eval_{};
  chess::move played_{chess::move::null()};
  chess::move killer_{chess::move::null()};
  std::array<chess::move, safe_depth_> pv_{};

  stack_entry(){ pv_.fill(chess::move::null()); }
};

struct stack{
  chess::position_history past_;
  chess::board present_;
  std::array<stack_entry, safe_depth_> future_{};
  
  stack_entry& at_(const depth_type& height){
    return future_[height];
  }

  chess::board root_pos() const { return present_; }

  size_t occurrences(const size_t& height, const zobrist::hash_type& hash) const {
    size_t occurrences_{0};
    for(auto it = future_.cbegin(); it != (future_.cbegin() + height); ++it){
      occurrences_ += static_cast<size_t>(it -> hash_ == hash);
    }
    return occurrences_ + past_.occurrences(hash);
  }
  
  std::string pv_string() const {
    auto bd = present_;
    std::string result{};
    for(const auto& pv_mv : future_[0].pv_){
      if(!bd.generate_moves().has(pv_mv)){ break; }
      result += pv_mv.name(bd.turn()) + " ";
      bd = bd.forward(pv_mv);
    }
    return result;
  }

  stack& clear_future(){
    future_.fill(stack_entry{});
    return *this;
  }

  stack(const chess::position_history& past, const chess::board& present) : past_{past}, present_{present} {}
};

struct stack_view{
  stack* view_;
  depth_type height_{};

  constexpr score_type effective_mate_score() const {
    return mate_score + height_;
  }
  
  chess::board root_pos() const { return view_ -> root_pos(); }

  bool is_two_fold(const zobrist::hash_type& hash) const {
    return view_ -> occurrences(height_, hash) >= 1;
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
  
  const std::array<chess::move, safe_depth_>& pv() const {
    return (view_ -> at_(height_)).pv_;
  }

  bool nmp_valid() const {
    return !counter().is_null() && !follow().is_null();
  }
  
  bool improving() const {
    return (height_ > 1) && (view_ -> at_(height_ - 2)).eval_ < (view_ -> at_(height_)).eval_;
  }

  const stack_view& set_hash(const zobrist::hash_type& hash) const {
    (view_ -> at_(height_)).hash_ = hash;
    return *this;
  }

  const stack_view& set_eval(const search::score_type& eval) const {
    (view_ -> at_(height_)).eval_ = eval;
    return *this;
  }

  const stack_view& set_played(const chess::move& played) const {
    (view_ -> at_(height_)).played_ = played;
    return *this;
  }

  const stack_view& prepend_to_pv(const chess::move& pv_mv) const {
    auto& our_pv = (view_ -> at_(height_)).pv_;
    const auto& child_pv = next().pv();
    our_pv[0] = pv_mv;
    const auto output_iter = our_pv.begin() + 1;
    std::copy(child_pv.cbegin(), child_pv.cbegin() + std::distance(output_iter, our_pv.end()), output_iter);
    return *this;
  }

  const stack_view& set_killer(const chess::move& killer) const {
    (view_ -> at_(height_)).killer_ = killer;
    return *this;
  }

  stack_view prev() const { return stack_view(view_, height_ - 1); }
  
  stack_view next() const { return stack_view(view_, height_ + 1); }
  
  stack_view(stack* view, const depth_type& height) : 
    view_{view},
    height_{std::min(safe_depth_ - 1, height)} 
  {
    assert((height >= 0));
  }
  
  static stack_view root(stack& st){ return stack_view(&st, 0); }
};


}

