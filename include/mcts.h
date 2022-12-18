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
#include <move.h>

#include <cstdint>
#include <mutex>
#include <random>
#include <tuple>
#include <vector>

namespace mcts {

using probability_type = float;
using index_type = std::uint8_t;
using node_count_type = std::uint64_t;

template <typename T>
struct index_map {
  std::vector<index_type> indices_{};
  std::vector<T> values_{};

  template <typename... Ts>
  void insert(const index_type& index, Ts... value) {
    indices_.push_back(index);
    values_.emplace_back(value...);
  }

  bool has_index(const index_type& index) { return std::find(indices_.begin(), indices_.end(), index) != indices_.end(); }

  std::vector<T>::iterator value_iter(const index_type& index) {
    const auto i_iter = std::find(indices_.begin(), indices_.end(), index);
    auto v_iter = values_.begin();
    std::advance(v_iter, std::distance(indices_.begin(), i_iter));
    return v_iter;
  }

  template <typename F>
  void visit(F&& f) const {
    for (auto [i_iter, v_iter] = std::pair(indices_.begin(), values_.begin()); i_iter != indices_.end(); ++i_iter, ++v_iter) { f(*i_iter, *v_iter); }
  }
};

struct tree_node_descriptor {
  probability_type q_value;
  node_count_type node_count;

  void update(const probability_type& win_percentage) {
    q_value = (q_value * node_count + win_percentage) / (node_count + 1);
    ++node_count;
  }

  static constexpr tree_node_descriptor from_win_percentage(const probability_type& win_percentage) {
    return tree_node_descriptor{win_percentage, static_cast<node_count_type>(1)};
  }
};

struct tree_node {
  std::mutex mutex_{};

  tree_node* parent_;
  index_type node_index_;
  std::vector<probability_type> policy_;

  index_map<tree_node> children_{};
  index_map<tree_node_descriptor> child_descriptors_{};

  void backup_(const index_type& index, const probability_type& value) {
    {
      std::lock_guard lock(mutex_);
      child_descriptors_.value_iter(index)->update(value);
    }

    if (parent_ != nullptr) { parent_->backup_(node_index_, value); }
  }

  void insert(const index_type& index, const probability_type& win_percentage, const std::vector<probability_type>& policy) {
    {
      std::lock_guard lock(mutex_);
      if (child_descriptors_.has_index(index)) { return; }

      child_descriptors_.insert(index, tree_node_descriptor::from_win_percentage(win_percentage));
      children_.insert(index, this, index, policy);
    }

    parent_->backup_(node_index_, win_percentage);
  }

  template <typename G>
  std::tuple<chess::board, tree_node*> traverse(G& generator, const chess::board& bd) {
    std::lock_guard lock(mutex_);
    const chess::move_list moves = bd.generate_moves<chess::generation_mode::all>();
    
    
    std::uniform_int_distribution<index_type> policy_distribution();


  }

  tree_node(tree_node* parent, const index_type& node_index, const std::vector<probability_type>& policy)
      : parent_{parent}, node_index_{node_index}, policy_{policy} {}
};

}  // namespace mcts