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

#include <bit_range.h>
#include <history_heuristic.h>
#include <move.h>

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <tuple>

namespace search {

constexpr std::uint32_t make_positive(const std::int32_t& x) {
  constexpr std::uint32_t upper = static_cast<std::uint32_t>(1) + std::numeric_limits<std::int32_t>::max();
  return upper + x;
}

enum class stage : std::uint8_t { transposition_table_move, noisy_and_check_moves, quiet_moves };

struct move_orderer_data {
  chess::move killer{chess::move::null()};
  chess::move follow{chess::move::null()};
  chess::move counter{chess::move::null()};
  chess::move first{chess::move::null()};

  chess::square_set threatened{};

  const chess::board* bd;
  const history_heuristic* hh;

  move_orderer_data& set_killer(const chess::move& mv) {
    killer = mv;
    return *this;
  }

  move_orderer_data& set_follow(const chess::move& mv) {
    follow = mv;
    return *this;
  }

  move_orderer_data& set_counter(const chess::move& mv) {
    counter = mv;
    return *this;
  }

  move_orderer_data& set_first(const chess::move& mv) {
    first = mv;
    return *this;
  }

  move_orderer_data& set_threatened(const chess::square_set& mask) {
    threatened = mask;
    return *this;
  }

  move_orderer_data(const chess::board* bd_, const history_heuristic* hh_) : bd{bd_}, hh{hh_} {}
};

struct move_orderer_entry {
  using value_ = bit::range<std::uint32_t, 0, 32>;
  using killer_ = bit::next_flag<value_>;
  using positive_noisy_ = bit::next_flag<killer_>;
  using first_ = bit::next_flag<positive_noisy_>;

  chess::move mv;
  std::uint64_t data_;

  const std::uint64_t& sort_key() const { return data_; }
  bool is_positive_noisy() const { return positive_noisy_::get(data_); }

  move_orderer_entry() = default;
  move_orderer_entry(const chess::move& mv_, bool is_positive_noisy, bool is_killer, const std::int32_t& value) : mv{mv_}, data_{0} {
    positive_noisy_::set(data_, is_positive_noisy);
    killer_::set(data_, is_killer);
    value_::set(data_, make_positive(value));
  }

  static inline move_orderer_entry make_noisy(const chess::move& mv, const bool positive_noisy, const std::int32_t& history_value) {
    return move_orderer_entry(mv, positive_noisy, false, positive_noisy ? mv.mvv_lva_key<std::int32_t>() : history_value);
  }

  static inline move_orderer_entry make_quiet(const chess::move& mv, const chess::move& killer, const std::int32_t& history_value) {
    return move_orderer_entry(mv, false, mv == killer, history_value);
  }
};

struct move_orderer_stepper {
  using entry_array_type = std::array<move_orderer_entry, chess::move_list::max_branching_factor>;

  entry_array_type entries_;
  entry_array_type::iterator begin_;
  entry_array_type::iterator end_;

  void update_list() {
    if (begin_ == end_) { return; }
    auto comparator = [](const move_orderer_entry& a, const move_orderer_entry& b) { return a.sort_key() < b.sort_key(); };
    std::iter_swap(begin_, std::max_element(begin_, end_, comparator));
  }

  chess::move current_move() const { return begin_->mv; }
  const move_orderer_entry& current_entry() const { return *begin_; }

  bool empty() const { return begin_ == end_; }
  void step() { ++begin_; }

  void add_move(const move_orderer_data& data, const chess::move& mv) {
    const history::context ctxt{data.follow, data.counter, data.threatened};

    *end_ = [&] {
      if (mv.is_noisy()) { return move_orderer_entry::make_noisy(mv, data.bd->see_gt(mv, 0), data.hh->compute_value(ctxt, mv)); }
      return move_orderer_entry::make_quiet(mv, data.killer, data.hh->compute_value(ctxt, mv));
    }();

    ++end_;
  }

  void add_moves(const move_orderer_data& data, const chess::move_list& list, const chess::move& excluded = chess::move::null()) {
    for (const chess::move& mv : list) {
      if (mv == excluded) { continue; }
      add_move(data, mv);
    }
  }

  move_orderer_stepper() : begin_{entries_.begin()}, end_{entries_.begin()} {}

  move_orderer_stepper& operator=(const move_orderer_stepper& other) = delete;
  move_orderer_stepper& operator=(move_orderer_stepper&& other) = delete;
  move_orderer_stepper(const move_orderer_stepper& other) = delete;
  move_orderer_stepper(move_orderer_stepper&& other) = delete;
};

struct move_orderer_iterator_end_tag {};

template <typename mode>
struct move_orderer_iterator {
  using difference_type = std::ptrdiff_t;
  using value_type = std::tuple<int, chess::move>;
  using pointer = std::tuple<int, chess::move>*;
  using reference = std::tuple<int, chess::move>&;
  using iterator_category = std::output_iterator_tag;

  static constexpr stage final_stage = stage::quiet_moves;

  stage current_stage_{stage::transposition_table_move};
  int current_idx_{0};
  move_orderer_stepper stepper_{};
  move_orderer_data data_;

  void update_stage() {
    switch (current_stage_) {
      case stage::transposition_table_move: {
        if (!stepper_.empty()) { break; }

        using stage_mode = chess::generation_mode::noisy_and_check::and_type<mode>;
        if constexpr (stage_mode::any) { stepper_.add_moves(data_, data_.bd->generate_moves<stage_mode>(), data_.first); }
        current_stage_ = stage::noisy_and_check_moves;
        [[fallthrough]];
      }

      case stage::noisy_and_check_moves: {
        if (!stepper_.empty() && stepper_.current_entry().is_positive_noisy()) { break; }

        using stage_mode = typename chess::generation_mode::quiet::and_type<mode>;
        if constexpr (stage_mode::any) { stepper_.add_moves(data_, data_.bd->generate_moves<stage_mode>(), data_.first); }
        current_stage_ = stage::quiet_moves;
        [[fallthrough]];
      }

      case stage::quiet_moves:
      default: {
        current_stage_ = stage::quiet_moves;
      }
    }
  }

  std::tuple<int, chess::move> operator*() const { return std::tuple(current_idx_, stepper_.current_move()); }

  move_orderer_iterator& operator++() {
    stepper_.step();
    update_stage();
    stepper_.update_list();
    ++current_idx_;
    return *this;
  }

  bool operator==(const move_orderer_iterator<mode>&) const { return false; }
  bool operator==(const move_orderer_iterator_end_tag&) const { return (current_stage_ == final_stage) && stepper_.empty(); }

  template <typename T>
  bool operator!=(const T& other) const {
    return !(*this == other);
  }

  move_orderer_iterator(const move_orderer_data& data) : data_{data} {
    if (!data.first.is_null() && data.bd->is_legal<mode>(data.first)) { stepper_.add_move(data, data.first); }
    update_stage();
    stepper_.update_list();
  }
};

template <typename mode>
struct move_orderer {
  using iterator = move_orderer_iterator<mode>;

  move_orderer_data data_;

  move_orderer_iterator<mode> begin() { return move_orderer_iterator<mode>(data_); }
  move_orderer_iterator_end_tag end() { return move_orderer_iterator_end_tag(); }

  move_orderer& set_first(const chess::move& mv) {
    data_.set_first(mv);
    return *this;
  }

  move_orderer(const move_orderer_data& data) : data_{data} {}
};

}  // namespace search