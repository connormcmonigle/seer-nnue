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

#include <apply.h>
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

namespace history {

using value_type = search::counter_type;

namespace constants {

inline constexpr size_t num_squares = 64;
inline constexpr size_t num_pieces = 6;

};  // namespace constants

struct context {
  move follow;
  move counter;
};

value_type formula(const value_type& x, const value_type& gain) {
  constexpr value_type history_multiplier = 32;
  constexpr value_type history_divisor = 512;
  return (gain * history_multiplier) - (x * std::abs(gain) / history_divisor);
}

struct butterfly_info {
  static constexpr size_t N = constants::num_squares * constants::num_squares;

  static constexpr bool is_applicable(const context&, const move&) { return true; }

  static constexpr size_t compute_index(const context&, const move& mv) {
    const size_t from = static_cast<size_t>(mv.from().index());
    const size_t to = static_cast<size_t>(mv.to().index());
    return from * constants::num_squares + to;
  }
};

struct counter_info {
  static constexpr size_t N = constants::num_squares * constants::num_pieces * constants::num_squares * constants::num_pieces;

  static constexpr bool is_applicable(const context& ctxt, const move&) { return !ctxt.counter.is_null(); }

  static constexpr size_t compute_index(const context& ctxt, const move& mv) {
    const size_t p0 = static_cast<size_t>(ctxt.counter.piece());
    const size_t from0 = static_cast<size_t>(ctxt.counter.from().index());
    const size_t p1 = static_cast<size_t>(mv.piece());
    const size_t to1 = static_cast<size_t>(mv.to().index());
    return p0 * constants::num_squares * constants::num_pieces * constants::num_squares + from0 * constants::num_pieces * constants::num_squares +
           p1 * constants::num_squares + to1;
  }
};

struct follow_info {
  static constexpr size_t N = constants::num_squares * constants::num_pieces * constants::num_squares * constants::num_pieces;

  static constexpr bool is_applicable(const context& ctxt, const move&) { return !ctxt.follow.is_null(); }

  static constexpr size_t compute_index(const context& ctxt, const move& mv) {
    const size_t p0 = static_cast<size_t>(ctxt.follow.piece());
    const size_t from0 = static_cast<size_t>(ctxt.follow.from().index());
    const size_t p1 = static_cast<size_t>(mv.piece());
    const size_t to1 = static_cast<size_t>(mv.to().index());
    return p0 * constants::num_squares * constants::num_pieces * constants::num_squares + from0 * constants::num_pieces * constants::num_squares +
           p1 * constants::num_squares + to1;
  }
};

template <typename T>
struct table {
  std::array<value_type, T::N> data_{};

  constexpr bool is_applicable(const context& ctxt, const move& mv) const { return T::is_applicable(ctxt, mv); }

  constexpr const value_type& at(const context& ctxt, const move& mv) const { return data_[T::compute_index(ctxt, mv)]; }
  constexpr value_type& at(const context& ctxt, const move& mv) { return data_[T::compute_index(ctxt, mv)]; }

  constexpr void clear() { data_.fill(value_type{}); }
};

template <typename... Ts>
struct combined {
  std::tuple<table<Ts>...> tables_{};

  constexpr combined<Ts...>& update(const context& ctxt, const move& best_move, const move_list& tried, const search::depth_type& depth) {
    constexpr value_type history_max = 400;

    auto single_update = [&, this](const auto& mv, const value_type& gain) {
      util::apply(tables_, [=](auto& tbl) {
        if (tbl.is_applicable(ctxt, mv)) { tbl.at(ctxt, mv) += formula(tbl.at(ctxt, mv), gain); }
      });
    };

    const value_type gain = std::min(history_max, depth * depth);
    std::for_each(tried.begin(), tried.end(), [single_update, gain](const move& mv) { single_update(mv, -gain); });
    single_update(best_move, gain);

    return *this;
  }

  constexpr void clear() {
    util::apply(tables_, [](auto& tbl) { tbl.clear(); });
  }

  constexpr value_type compute_value(const context& ctxt, const move& mv) const {
    value_type result{};
    util::apply(tables_, [&](const auto& tbl) {
      if (tbl.is_applicable(ctxt, mv)) { result += tbl.at(ctxt, mv); }
    });
    return result;
  }
};

}  // namespace history

using history_heuristic = history::combined<history::butterfly_info, history::counter_info, history::follow_info>;

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
