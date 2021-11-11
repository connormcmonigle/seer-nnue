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
#include <tuple>

namespace chess {

constexpr std::uint32_t make_positive(const std::int32_t& x) {
  constexpr std::uint32_t upper = static_cast<std::uint32_t>(1) + std::numeric_limits<std::int32_t>::max();
  return upper + x;
}

struct move_orderer_data {
  move killer{move::null()};
  move follow{move::null()};
  move counter{move::null()};
  move first{move::null()};

  const board* bd;
  const move_list* list;
  const history_heuristic* hh;

  move_orderer_data& set_killer(const move& mv) {
    killer = mv;
    return *this;
  }

  move_orderer_data& set_follow(const move& mv) {
    follow = mv;
    return *this;
  }

  move_orderer_data& set_counter(const move& mv) {
    counter = mv;
    return *this;
  }

  move_orderer_data& set_first(const move& mv) {
    first = mv;
    return *this;
  }

  move_orderer_data(const board* bd_, const move_list* list_, const history_heuristic* hh_) : bd{bd_}, list{list_}, hh{hh_} {}
};

struct move_orderer_entry {
  using value_ = bit::range<std::uint32_t, 0, 32>;
  using killer_ = bit::next_flag<value_>;
  using nonnegative_noisy_ = bit::next_flag<killer_>;
  using first_ = bit::next_flag<nonnegative_noisy_>;

  move mv;
  std::uint64_t data;

  move_orderer_entry() = default;
  move_orderer_entry(const move& mv_, bool is_first, bool is_noisy, bool is_killer, std::int32_t value) : mv{mv_}, data{0} {
    first_::set(data, is_first);
    nonnegative_noisy_::set(data, is_noisy);
    killer_::set(data, is_killer);
    value_::set(data, make_positive(value));
  }
};

struct move_orderer_iterator_end_tag {};

struct move_orderer_iterator {
  using difference_type = std::ptrdiff_t;
  using value_type = std::tuple<std::ptrdiff_t, move>;
  using pointer = std::tuple<std::ptrdiff_t, move>*;
  using reference = std::tuple<std::ptrdiff_t, move>&;
  using iterator_category = std::output_iterator_tag;

  using entry_array_type = std::array<move_orderer_entry, move_list::max_branching_factor>;

  entry_array_type entries_;
  entry_array_type::iterator begin_;
  entry_array_type::iterator end_;

  void update_list_() {
    std::iter_swap(begin_, std::max_element(begin_, end_, [](const move_orderer_entry& a, const move_orderer_entry& b) { return a.data < b.data; }));
  }

  move_orderer_iterator& operator++() {
    ++begin_;
    update_list_();
    return *this;
  }

  move_orderer_iterator operator++(int) {
    auto retval = *this;
    ++(*this);
    return retval;
  }

  constexpr bool operator==(const move_orderer_iterator&) const { return false; }
  constexpr bool operator==(const move_orderer_iterator_end_tag&) const { return begin_ == end_; }

  template <typename T>
  constexpr bool operator!=(const T& other) const {
    return !(*this == other);
  }

  std::tuple<std::ptrdiff_t, move> operator*() const { return std::tuple(begin_ - entries_.begin(), begin_->mv); }

  move_orderer_iterator(const move_orderer_data& data) : entries_{}, begin_{entries_.begin()} {
    end_ = std::transform(data.list->begin(), data.list->end(), entries_.begin(), [&data](const move& mv) {
      const bool quiet = mv.is_quiet();
      const std::int32_t value = quiet ? data.hh->compute_value(history::context{data.follow, data.counter}, mv) : data.bd->see<std::int32_t>(mv);
      return move_orderer_entry(mv, mv == data.first, !quiet && value >= 0, quiet && mv == data.killer, value);
    });

    update_list_();
  }
};

struct move_orderer {
  using iterator = move_orderer_iterator;

  move_orderer_data data_;

  move_orderer_iterator begin() { return move_orderer_iterator(data_); }
  move_orderer_iterator_end_tag end() { return move_orderer_iterator_end_tag(); }

  move_orderer& set_first(const move& mv) {
    data_.set_first(mv);
    return *this;
  }

  move_orderer(const move_orderer_data& data) : data_{data} {}
};

}  // namespace chess
