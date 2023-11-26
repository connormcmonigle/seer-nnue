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
#include <search_constants.h>
#include <search_worker.h>
#include <transposition_table.h>

#include <cstdint>
#include <memory>
#include <mutex>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

namespace mcts {

using probability_type = float;
using real_type = float;
using index_type = std::uint8_t;
using node_count_type = std::uint64_t;

namespace parameters {

constexpr real_type cpuct_base = 19652;
constexpr real_type cpuct_init = 1.25;
constexpr probability_type loss_value = 0.0;
constexpr probability_type draw_value = 0.5;
constexpr probability_type win_value = 1.0;
constexpr search::depth_type ab_search_start_depth = 1;
constexpr search::depth_type ab_search_depth_limit = 7;

}  // namespace parameters

constexpr probability_type complement(const probability_type& probability) {
  constexpr probability_type one = static_cast<probability_type>(1);
  return one - probability;
}

std::optional<probability_type> exact_outcome(const chess::position_history& history, const chess::board& state) {
  const bool is_loss = state.is_check() && state.generate_moves<chess::generation_mode::all>().empty();
  const bool is_draw = state.is_trivially_drawn() || history.is_two_fold(state.hash()) ||
                       (!state.is_check() && state.generate_moves<chess::generation_mode::all>().empty()) ||
                       (state.is_rule50_draw() && (!state.is_check() || !state.generate_moves<chess::generation_mode::all>().empty()));
  constexpr bool is_win = false;

  return is_win  ? std::optional(parameters::win_value) :
         is_draw ? std::optional(parameters::draw_value) :
         is_loss ? std::optional(parameters::loss_value) :
                   std::nullopt;
}

template <typename T>
struct index_map {
  std::vector<index_type> indices_{};
  std::vector<T> values_{};

  const std::vector<index_type>& indices() const { return indices_; }
  const std::vector<T>& values() const { return values_; }

  template <typename... Ts>
  void insert(const index_type& index, Ts&&... value) {
    indices_.push_back(index);
    values_.emplace_back(std::forward<Ts>(value)...);
  }

  bool has_index(const index_type& index) const { return std::find(indices_.begin(), indices_.end(), index) != indices_.end(); }

  T& value(const index_type& index) {
    const auto i_iter = std::find(indices_.begin(), indices_.end(), index);
    auto v_iter = values_.begin();
    std::advance(v_iter, std::distance(indices_.begin(), i_iter));
    return *v_iter;
  }

  template <typename F>
  void visit(F&& f) const {
    for (auto [i_iter, v_iter] = std::pair(indices_.begin(), values_.begin()); i_iter != indices_.end(); ++i_iter, ++v_iter) { f(*i_iter, *v_iter); }
  }
};

struct prior_estimate {
  probability_type expected_outcome;
  std::vector<probability_type> policy;
};

struct tree_node_descriptor {
  probability_type q_value_;
  node_count_type node_count_;

  void update(const probability_type& expected_outcome) {
    const real_type one = static_cast<real_type>(1);
    const real_type node_count_value = static_cast<real_type>(node_count_);
    q_value_ = (q_value_ * node_count_value + expected_outcome) / (node_count_value + one);
    ++node_count_;
  }

  const real_type& q_value() const { return q_value_; }
  const node_count_type& node_count() const { return node_count_; }

  real_type u_value(const real_type& cpuct_value, const probability_type& policy_value, const node_count_type& parent_node_count) const {
    constexpr real_type one = static_cast<real_type>(1);
    const real_type parent_node_count_value = static_cast<real_type>(parent_node_count);
    const real_type node_count_value = static_cast<real_type>(node_count_);
    return cpuct_value * policy_value * std::sqrt(parent_node_count_value) / (one + node_count_value);
  }

  static real_type compute_cpuct_value(const node_count_type& parent_node_count) {
    constexpr real_type one = static_cast<real_type>(1);
    const real_type parent_node_count_value = static_cast<real_type>(parent_node_count);
    return std::log((one + parent_node_count_value + parameters::cpuct_base) / parameters::cpuct_base) + parameters::cpuct_init;
  }

  static real_type uninit_u_value(const real_type& cpuct_value, const probability_type& policy_value, const node_count_type& parent_node_count) {
    const real_type parent_node_count_value = static_cast<real_type>(parent_node_count);
    return cpuct_value * policy_value * std::sqrt(parent_node_count_value);
  }

  static constexpr probability_type uninit_q_value() { return parameters::loss_value; }

  static constexpr tree_node_descriptor from_expected_outcome(const probability_type& expected_outcome) {
    return tree_node_descriptor{expected_outcome, static_cast<node_count_type>(1)};
  }
};

struct tree_node {
  tree_node* parent_;
  index_type node_index_;
  std::vector<probability_type> policy_;

  index_map<std::unique_ptr<tree_node>> children_{};
  index_map<tree_node_descriptor> child_descriptors_{};

  void backup(const index_type& index, const probability_type& value) {
    child_descriptors_.value(index).update(value);
    if (parent_ != nullptr) { parent_->backup(node_index_, complement(value)); }
  }

  void insert(const index_type& index, const prior_estimate& estimate) {
    child_descriptors_.insert(index, tree_node_descriptor::from_expected_outcome(complement(estimate.expected_outcome)));
    children_.insert(index, std::make_unique<tree_node>(this, index, estimate.policy));
    if (parent_ != nullptr) { parent_->backup(node_index_, estimate.expected_outcome); }
  }

  bool has_index(const index_type& index) const { return children_.has_index(index); }
  tree_node* next(const index_type& index) { return children_.value(index).get(); }

  index_type select_index() const {
    const node_count_type parent_node_count = std::accumulate(
                                                  child_descriptors_.values().begin(), child_descriptors_.values().end(), node_count_type{},
                                                  [](const node_count_type& total_node_count, const tree_node_descriptor& descriptor) {
                                                    return descriptor.node_count() + total_node_count;
                                                  }) +
                                              static_cast<node_count_type>(1);

    const real_type cpuct_value = tree_node_descriptor::compute_cpuct_value(parent_node_count);

    std::vector<real_type> selection_weights{};
    std::transform(
        policy_.begin(), policy_.end(), std::back_inserter(selection_weights),
        [&parent_node_count, &cpuct_value](const probability_type& policy_value) {
          return tree_node_descriptor::uninit_q_value() + tree_node_descriptor::uninit_u_value(cpuct_value, policy_value, parent_node_count);
        });

    child_descriptors_.visit(
        [this, &parent_node_count, &cpuct_value, &selection_weights](const index_type& index, const tree_node_descriptor& descriptor) {
          selection_weights[index] = descriptor.q_value() + descriptor.u_value(cpuct_value, policy_[index], parent_node_count);
        });

    const auto iter = std::max_element(selection_weights.begin(), selection_weights.end());
    return std::distance(selection_weights.begin(), iter);
  }

  index_type best_index() const {
    index_type best{};
    node_count_type max_node_count{};

    child_descriptors_.visit([&](const index_type& index, const tree_node_descriptor& descriptor) {
      if (descriptor.node_count() >= max_node_count) {
        best = index;
        max_node_count = descriptor.node_count();
      }
    });

    return best;
  }

  probability_type q_value() const {
    const auto iter = std::max_element(child_descriptors_.values().begin(), child_descriptors_.values().end(), [](const auto& a, const auto& b) {
      return a.q_value() < b.q_value();
    });

    return iter->q_value();
  }

  tree_node(tree_node* parent, const index_type& node_index, const std::vector<probability_type>& policy)
      : parent_{parent}, node_index_{node_index}, policy_{policy} {}
};

struct tree_walker {
  search::search_worker worker_;

  prior_estimate compute_prior_estimate(const chess::position_history& history, const chess::board& state) {
    if (const std::optional<probability_type> outcome = exact_outcome(history, state); outcome.has_value()) {
      return prior_estimate{outcome.value(), {}};
    }

    worker_.go(history, state, parameters::ab_search_start_depth);
    worker_.iterative_deepening_loop();
    return prior_estimate{worker_.expected_outcome<probability_type>(), worker_.policy<probability_type>()};
  }

  void walk(chess::position_history history, chess::board state, tree_node* tree) {
    for (;;) {
      const index_type index = tree->select_index();
      history.push_(state.hash());
      const chess::move_list actions = state.generate_moves<chess::generation_mode::all>();
      state = state.forward(actions[index]);

      if (!tree->has_index(index)) {
        tree->insert(index, compute_prior_estimate(history, state));
        break;
      }

      if (const std::optional<probability_type> outcome = exact_outcome(history, state); outcome.has_value()) {
        const probability_type outcome_value = outcome.value();
        tree->backup(index, complement(outcome_value));
        break;
      }

      tree = tree->next(index);
    }
  }

  tree_walker(const nnue::weights* weights, std::shared_ptr<search::transposition_table> tt, std::shared_ptr<search::search_constants> constants)
      : worker_(weights, tt, constants, [this](const auto& worker) {
          if (worker.depth() >= parameters::ab_search_depth_limit) { worker_.stop(); }
        }) {}
};

}  // namespace mcts