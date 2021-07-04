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

#include <enum_util.h>
#include <move.h>
#include <position_history.h>
#include <search_constants.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <iostream>
#include <type_traits>

namespace chess {



struct history_heuristic {
  static constexpr size_t num_squares = 64;
  static constexpr size_t num_pieces = 6;

  struct components {
    search::counter_type follow;
    search::counter_type counter;
    search::counter_type butterfly;
  };

  std::array<search::counter_type, num_squares * num_squares> butterfly_{};
  std::array<search::counter_type, num_pieces * num_squares * num_pieces * num_squares> counter_{};
  std::array<search::counter_type, num_pieces * num_squares * num_pieces * num_squares> follow_{};

  size_t butterfly_idx_(const move& mv) const {
    const size_t from = static_cast<size_t>(mv.from().index());
    const size_t to = static_cast<size_t>(mv.to().index());
    return from * num_squares + to;
  }

  size_t counter_idx_(const move& them_mv, const move& mv) const {
    const size_t p0 = static_cast<size_t>(them_mv.piece());
    const size_t from0 = static_cast<size_t>(them_mv.from().index());
    const size_t p1 = static_cast<size_t>(mv.piece());
    const size_t to1 = static_cast<size_t>(mv.to().index());
    return p0 * num_squares * num_pieces * num_squares + from0 * num_pieces * num_squares + p1 * num_squares + to1;
  }

  size_t follow_idx_(const move& us_mv, const move& mv) const {
    const size_t p0 = static_cast<size_t>(us_mv.piece());
    const size_t from0 = static_cast<size_t>(us_mv.from().index());
    const size_t p1 = static_cast<size_t>(mv.piece());
    const size_t to1 = static_cast<size_t>(mv.to().index());
    return p0 * num_squares * num_pieces * num_squares + from0 * num_pieces * num_squares + p1 * num_squares + to1;
  }

  history_heuristic& clear() {
    butterfly_.fill(search::counter_type{});
    counter_.fill(search::counter_type{});
    follow_.fill(search::counter_type{});
    return *this;
  }

  history_heuristic& update(const move& follow, const move& counter, const move& best_move, const move_list& tried, const search::depth_type& depth) {
    // more or less lifted from ethereal
    constexpr search::counter_type history_max = 400;
    constexpr search::counter_type history_multiplier = 32;
    constexpr search::counter_type history_divisor = 512;
    assert((!tried.has(best_move)));
    auto single_update = [&, this](const auto& mv, const search::counter_type& gain) {
      auto formula = [=](const search::counter_type& x) { return (gain * history_multiplier) - (x * std::abs(gain) / history_divisor); };
      // update butterfly history
      {
        const size_t idx = butterfly_idx_(mv);
        butterfly_[idx] += formula(butterfly_[idx]);
      }
      // update counter move history
      {
        const size_t idx = counter_idx_(counter, mv);
        counter_[idx] += formula(counter_[idx]);
      }
      // update follow up move history
      {
        const size_t idx = follow_idx_(follow, mv);
        follow_[idx] += formula(follow_[idx]);
      }
    };
    // limit gain to prevent saturation
    const search::counter_type gain = std::min(history_max, static_cast<search::counter_type>(depth) * static_cast<search::counter_type>(depth));
    std::for_each(tried.begin(), tried.end(), [single_update, gain](const move& mv) { single_update(mv, -gain); });
    single_update(best_move, gain);
    return *this;
  }

  components compute_components(const move& follow, const move& counter, const move& mv) {
    return components{follow_[follow_idx_(follow, mv)], counter_[counter_idx_(counter, mv)], butterfly_[butterfly_idx_(mv)]};
  }

  search::counter_type compute_value(const move& follow, const move& counter, const move& mv) const {
    return follow_[follow_idx_(follow, mv)] + counter_[counter_idx_(counter, mv)] + butterfly_[butterfly_idx_(mv)];
  }

};

struct sided_history_heuristic : sided<sided_history_heuristic, history_heuristic> {
  history_heuristic white;
  history_heuristic black;

  sided_history_heuristic& clear() {
    white.clear();
    black.clear();
    return *this;
  }

  sided_history_heuristic() : white{}, black{} {}
};

}  // namespace chess
