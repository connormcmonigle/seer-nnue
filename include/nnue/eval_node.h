/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
  the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not,
  see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <chess/board.h>
#include <chess/move.h>
#include <nnue/eval.h>
#include <nnue/feature_reset_cache.h>

namespace nnue {

struct eval_node {
  struct context {
    sided_feature_reset_cache* reset_cache_{nullptr};
    eval_node* parent_node_{nullptr};
    const chess::board* parent_board_{nullptr};
    const chess::move move_{chess::move::null()};
  };

  bool dirty_;

  union {
    context context_;
    eval eval_;
  } data_;

  [[nodiscard]] bool dirty() const noexcept { return dirty_; }
  [[nodiscard]] eval dirty_evaluator() const noexcept { return !dirty_ ? data_.eval_ : data_.context_.parent_node_->dirty_evaluator().next_child(); }

  [[nodiscard]] const eval& evaluator() {
    if (!dirty_) { return data_.eval_; }

    dirty_ = false;
    const context ctxt = data_.context_;

    if (ctxt.parent_board_->requires_feature_reset(ctxt.move_)) {
      data_.eval_ = ctxt.parent_node_->dirty_evaluator().next_child();
      ctxt.parent_board_->forward(ctxt.move_).full_feature_reset_with_cache(*ctxt.reset_cache_, data_.eval_);
    } else {
      data_.eval_ = ctxt.parent_node_->evaluator().next_child();
      ctxt.parent_board_->feature_move_delta(ctxt.move_, *ctxt.reset_cache_, data_.eval_);
    }

    return data_.eval_;
  }

  [[nodiscard]] eval_node dirty_child(sided_feature_reset_cache* reset_cache, const chess::board* bd, const chess::move& mv) noexcept {
    return eval_node::dirty_node(context{reset_cache, this, bd, mv});
  }

  [[nodiscard]] static eval_node dirty_node(const context& context) noexcept { return eval_node{true, {context}}; }

  [[nodiscard]] static eval_node clean_node(const eval& eval) noexcept {
    eval_node result{};
    result.dirty_ = false;
    result.data_.eval_ = eval;
    return result;
  }
};

}  // namespace nnue
