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
#include <feature_transformer_cache.h>
#include <nnue_model.h>

#include <optional>

namespace nnue {

struct eval_node {
  struct context {
    eval_node* parent_node_{nullptr};
    feature_transformer_cache* cache{nullptr};
    const chess::board* parent_board_{nullptr};
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
    const bool refresh_required = ctxt.parent_board_->requires_feature_refresh(ctxt.move_);
    const zobrist::hash_type hash_after = ctxt.parent_board_->guess_hash_after(ctxt.move_);

    const auto cached_state = [&, this]() -> std::optional<eval::base_type> {
      if (refresh_required) { return ctxt.cache->find(hash_after); }
      return std::nullopt;
    }();

    if (cached_state.has_value()) {
      data_.eval_ = ctxt.parent_node_->evaluator().next_child_from_cached_state(*cached_state);
    } else {
      data_.eval_ = ctxt.parent_node_->evaluator().next_child();
      ctxt.parent_board_->feature_move_delta(ctxt.move_, data_.eval_);
      if (refresh_required) { ctxt.cache->insert(hash_after, data_.eval_.base()); }
    }

    return data_.eval_;
  }

  eval_node dirty_child(feature_transformer_cache* cache, const chess::board* bd, const chess::move& mv) {
    return eval_node::dirty_node(context{this, cache, bd, mv});
  }

  static eval_node dirty_node(const context& context) { return eval_node{true, {context}}; }

  static eval_node clean_node(const eval& eval) {
    eval_node result{};
    result.dirty_ = false;
    result.data_.eval_ = eval;
    return result;
  }
};

}  // namespace nnue