/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

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

#include <board.h>
#include <chess_types.h>
#include <evaluator_cache.h>
#include <feature_util.h>
#include <nnue_model.h>
#include <nnue_util.h>
#include <search_constants.h>
#include <weights_streamer.h>

#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>

namespace nnue {

struct eval_node {
  struct context {
    eval_node* parent_node_{nullptr};
    search::evaluator_cache* cache_{nullptr};

    const chess::board* parent_board_{nullptr};
    const chess::board* current_board_{nullptr};
    const chess::move move_{chess::move::null()};
  };

  bool dirty_;

  union {
    context context_;
    eval eval_;
  } data_;

  bool dirty() const { return dirty_; }

  const eval& evaluator() {
    if (!dirty_) { return data_.eval_; }

    dirty_ = false;
    const context ctxt = data_.context_;
    if (const auto key = ctxt.current_board_->hash(); ctxt.cache_->contains(key)) { return data_.eval_ = ctxt.cache_->get(key); }
    data_.eval_ = ctxt.parent_board_->apply_update(ctxt.move_, ctxt.parent_node_->evaluator());
    ctxt.cache_->insert(ctxt.current_board_->hash(), data_.eval_);
    return data_.eval_;
  }

  eval_node dirty_child(search::evaluator_cache* cache, const chess::board* parent, const chess::board* current, const chess::move& mv) {
    return eval_node::dirty_node(context{this, cache, parent, current, mv});
  }

  static eval_node dirty_node(const context& context) { return eval_node{true, context}; }

  static eval_node clean_node(const eval& eval) {
    eval_node result{};
    result.dirty_ = false;
    result.data_.eval_ = eval;
    return result;
  }
};

}  // namespace nnue