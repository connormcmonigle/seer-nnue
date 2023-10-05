/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

#include <chess/board.h>
#include <chess/board_history.h>
#include <chess/move.h>
#include <chess/types.h>
#include <search/search_constants.h>
#include <zobrist/util.h>

#include <algorithm>
#include <array>
#include <string>

namespace search {

struct stack_entry {
  zobrist::hash_type hash_{};
  score_type eval_{};

  chess::move played_{chess::move::null()};
  chess::move killer_{chess::move::null()};
  chess::move excluded_{chess::move::null()};

  std::array<chess::move, safe_depth> pv_{};

  stack_entry() noexcept { pv_.fill(chess::move::null()); }
};

struct search_stack {
  depth_type selective_depth_{0};

  chess::board_history past_;
  chess::board present_;
  std::array<stack_entry, safe_depth> future_{};

  [[nodiscard]] constexpr depth_type selective_depth() const noexcept { return selective_depth_; }
  [[nodiscard]] constexpr const chess::board& root() const noexcept { return present_; }

  [[nodiscard]] constexpr stack_entry& at(const depth_type& height) noexcept { return future_[height]; }

  [[maybe_unused]] constexpr search_stack& update_selective_depth(const depth_type& height) noexcept {
    selective_depth_ = std::max(selective_depth_, height);
    return *this;
  }

  [[nodiscard]] std::size_t count(const std::size_t& height, const zobrist::hash_type& hash) const noexcept;

  [[nodiscard]] std::string pv_string() const noexcept;
  [[nodiscard]] chess::move ponder_move() const noexcept;

  [[maybe_unused]] search_stack& clear_future() noexcept;

  search_stack(const chess::board_history& past, const chess::board& present) noexcept;
};

struct stack_view {
  search_stack* view_;
  depth_type height_{};

  [[nodiscard]] constexpr score_type loss_score() const noexcept { return mate_score + height_; }
  [[nodiscard]] constexpr score_type win_score() const noexcept { return -mate_score - height_; }

  [[nodiscard]] constexpr bool reached_max_height() const noexcept { return height_ >= (safe_depth - 1); }
  [[nodiscard]] constexpr depth_type height() const noexcept { return height_; }

  [[nodiscard]] constexpr const chess::board& root_position() const noexcept { return view_->root(); }

  [[nodiscard]] inline bool is_two_fold(const zobrist::hash_type& hash) const noexcept { return view_->count(height_, hash) >= 1; }

  [[nodiscard]] constexpr chess::move counter() const noexcept {
    if (height_ < 1) { return chess::move::null(); }
    return view_->at(height_ - 1).played_;
  }

  [[nodiscard]] constexpr chess::move follow() const noexcept {
    if (height_ < 2) { return chess::move::null(); }
    return view_->at(height_ - 2).played_;
  }

  [[nodiscard]] constexpr chess::move previous_follow() const noexcept {
    if (height_ < 4) { return chess::move::null(); }
    return view_->at(height_ - 4).played_;
  }

  [[nodiscard]] constexpr chess::move killer() const noexcept { return view_->at(height_).killer_; }

  [[nodiscard]] constexpr chess::move excluded() const noexcept { return view_->at(height_).excluded_; }

  [[nodiscard]] constexpr bool has_excluded() const noexcept { return !view_->at(height_).excluded_.is_null(); }

  [[nodiscard]] constexpr const std::array<chess::move, safe_depth>& pv() const { return view_->at(height_).pv_; }

  [[nodiscard]] constexpr bool nmp_valid() const noexcept { return !counter().is_null() && !follow().is_null(); }

  [[nodiscard]] constexpr bool improving() const noexcept { return (height_ >= 2) && view_->at(height_ - 2).eval_ < view_->at(height_).eval_; }

  [[maybe_unused]] constexpr const stack_view& set_hash(const zobrist::hash_type& hash) const noexcept {
    view_->at(height_).hash_ = hash;
    return *this;
  }

  [[maybe_unused]] constexpr const stack_view& set_eval(const score_type& eval) const noexcept {
    view_->at(height_).eval_ = eval;
    return *this;
  }

  [[maybe_unused]] constexpr const stack_view& set_played(const chess::move& played) const noexcept {
    view_->at(height_).played_ = played;
    return *this;
  }

  [[maybe_unused]] inline const stack_view& prepend_to_pv(const chess::move& pv_mv) const noexcept {
    const auto& child_pv = next().pv();
    auto output_iter = view_->at(height_).pv_.begin();
    *(output_iter++) = pv_mv;
    std::copy(child_pv.begin(), child_pv.begin() + std::distance(output_iter, view_->at(height_).pv_.end()), output_iter);
    return *this;
  }

  [[maybe_unused]] constexpr const stack_view& set_killer(const chess::move& killer) const noexcept {
    view_->at(height_).killer_ = killer;
    return *this;
  }

  [[maybe_unused]] constexpr const stack_view& set_excluded(const chess::move& excluded) const noexcept {
    view_->at(height_).excluded_ = excluded;
    return *this;
  }

  [[nodiscard]] constexpr stack_view prev() const noexcept { return stack_view(view_, height_ - 1); }
  [[nodiscard]] constexpr stack_view next() const noexcept { return stack_view(view_, height_ + 1); }

  constexpr stack_view(search_stack* view, const depth_type& height) : view_{view}, height_{height} { view_->update_selective_depth(height); }
  [[nodiscard]] static constexpr stack_view root(search_stack& st) noexcept { return stack_view(&st, 0); }
};

}  // namespace search
