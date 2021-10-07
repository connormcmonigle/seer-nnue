/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <board.h>
#include <enum_util.h>
#include <move.h>
#include <position_history.h>
#include <search_constants.h>
#include <zobrist_util.h>

#include <algorithm>
#include <array>
#include <string>

namespace search {

constexpr depth_type safe_depth_ = max_depth + max_depth_margin;

struct stack_entry {
  zobrist::hash_type hash_{};
  score_type eval_{};
  chess::move played_{chess::move::null()};
  chess::move killer_{chess::move::null()};
  chess::move excluded_{chess::move::null()};
  std::array<chess::move, safe_depth_> pv_{};

  stack_entry() { pv_.fill(chess::move::null()); }
};

struct stack {
  depth_type sel_depth_{0};

  chess::position_history past_;
  chess::board present_;
  std::array<stack_entry, safe_depth_> future_{};

  stack_entry& at(const depth_type& height) {
    sel_depth_ = std::max(sel_depth_, height);
    return future_[height];
  }

  chess::board root_pos() const { return present_; }
  depth_type sel_depth() const { return sel_depth_; }

  size_t occurrences(const size_t& height, const zobrist::hash_type& hash) const {
    size_t occurrences_{0};
    for (auto it = future_.cbegin(); it != (future_.cbegin() + height); ++it) { occurrences_ += static_cast<size_t>(it->hash_ == hash); }
    return occurrences_ + past_.occurrences(hash);
  }

  std::string pv_string() const {
    auto bd = present_;
    std::string result{};
    for (const auto& pv_mv : future_.begin()->pv_) {
      if (!bd.generate_moves().has(pv_mv)) { break; }
      result += pv_mv.name(bd.turn()) + " ";
      bd = bd.forward(pv_mv);
    }
    return result;
  }

  chess::move ponder_move() const { return *(future_.begin()->pv_.begin() + 1); }

  stack& clear_future() {
    sel_depth_ = 0;
    future_.fill(stack_entry{});
    return *this;
  }

  stack(const chess::position_history& past, const chess::board& present) : past_{past}, present_{present} {}
};

struct stack_view {
  stack* view_;
  depth_type height_{};

  constexpr score_type loss_score() const {  return mate_score + height_; }

  constexpr score_type win_score() const { return -mate_score - height_; }

  bool reached_max_height() const { return height_ >= (safe_depth_ - 1); }

  depth_type height() const { return height_; }

  chess::board root_pos() const { return view_->root_pos(); }

  bool is_two_fold(const zobrist::hash_type& hash) const { return view_->occurrences(height_, hash) >= 1; }

  chess::move counter() const {
    if (height_ <= 0) { return chess::move::null(); }
    return view_->at(height_ - 1).played_;
  }

  chess::move follow() const {
    if (height_ <= 1) { return chess::move::null(); }
    return view_->at(height_ - 2).played_;
  }

  chess::move killer() const { return view_->at(height_).killer_; }

  chess::move excluded() const { return view_->at(height_).excluded_; }

  bool has_excluded() const { return !view_->at(height_).excluded_.is_null(); }

  const std::array<chess::move, safe_depth_>& pv() const { return view_->at(height_).pv_; }

  bool nmp_valid() const { return !counter().is_null() && !follow().is_null(); }

  bool improving() const { return (height_ >= 2) && view_->at(height_ - 2).eval_ < view_->at(height_).eval_; }

  const stack_view& set_hash(const zobrist::hash_type& hash) const {
    view_->at(height_).hash_ = hash;
    return *this;
  }

  const stack_view& set_eval(const score_type& eval) const {
    view_->at(height_).eval_ = eval;
    return *this;
  }

  const stack_view& set_played(const chess::move& played) const {
    view_->at(height_).played_ = played;
    return *this;
  }

  const stack_view& prepend_to_pv(const chess::move& pv_mv) const {
    const auto& child_pv = next().pv();
    auto output_iter = view_->at(height_).pv_.begin();
    *(output_iter++) = pv_mv;
    std::copy(child_pv.begin(), child_pv.begin() + std::distance(output_iter, view_->at(height_).pv_.end()), output_iter);
    return *this;
  }

  const stack_view& set_killer(const chess::move& killer) const {
    view_->at(height_).killer_ = killer;
    return *this;
  }

  const stack_view& set_excluded(const chess::move& excluded) const {
    view_->at(height_).excluded_ = excluded;
    return *this;
  }

  stack_view prev() const { return stack_view(view_, height_ - 1); }

  stack_view next() const { return stack_view(view_, height_ + 1); }

  stack_view(stack* view, const depth_type& height) : view_{view}, height_{height} { assert((height >= 0)); }

  static stack_view root(stack& st) { return stack_view(&st, 0); }
};

}  // namespace search
