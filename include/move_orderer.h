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

template<typename T>
constexpr T mvv_lva(const move& mv) {
  constexpr T num_pieces = 6;
  return static_cast<T>(mv.captured()) * num_pieces + (num_pieces - static_cast<T>(mv.piece()));
}

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
  std::uint64_t data{0};

  move_orderer_entry(const move& mv_, bool is_first, bool is_noisy, bool is_killer, std::int32_t value) : mv{mv_} {
    first_::set(data, is_first);
    nonnegative_noisy_::set(data, is_noisy);
    killer_::set(data, is_killer);
    value_::set(data, make_positive(value));
  }

  move_orderer_entry(){};
};

struct move_orderer_iterator {
  using difference_type = long;
  using value_type = std::tuple<int, move>;
  using pointer = std::tuple<int, move>*;
  using reference = std::tuple<int, move>&;
  using iterator_category = std::output_iterator_tag;

  size_t move_count_{0};
  int idx_{0};
  std::array<move_orderer_entry, move_list::max_branching_factor> entries_{};

  void update_list_() {
    auto best = [this](const size_t& i0, const size_t& i1) { return (entries_[i0].data >= entries_[i1].data) ? i0 : i1; };

    size_t best_idx = idx_;
    for (size_t i(idx_); i < move_count_; ++i) { best_idx = best(best_idx, i); }

    std::swap(entries_[best_idx], entries_[idx_]);
  }

  move_orderer_iterator& operator++() {
    ++idx_;
    if (idx_ < static_cast<int>(move_count_)) { update_list_(); }
    return *this;
  }

  move_orderer_iterator operator++(int) {
    auto retval = *this;
    ++(*this);
    return retval;
  }

  bool operator==(const move_orderer_iterator& other) const { return other.idx_ == idx_; }

  bool operator!=(const move_orderer_iterator& other) const { return !(*this == other); }

  std::tuple<int, move> operator*() const { return std::tuple(idx_, entries_[idx_].mv); }

  move_orderer_iterator(const move_orderer_data& data, const int& idx) : move_count_{data.list->size()}, idx_{idx} {
    std::transform(data.list->begin(), data.list->end(), entries_.begin(), [&data](const move& mv) {
      const bool quiet = mv.is_quiet();
      const std::int32_t value = [&]{
        if (quiet) { return  data.hh->compute_value(history::context{data.follow, data.counter}, mv); }
        const search::see_type see_value = data.bd->see<search::see_type>(mv);
        return see_value >= 0 ? mvv_lva<search::see_type>(mv) : see_value;
      }();
      return move_orderer_entry(mv, mv == data.first, !quiet && value >= 0, quiet && mv == data.killer, value);
    });
    update_list_();
  }

  move_orderer_iterator(const int& idx) : idx_{idx} {}
};

struct move_orderer {
  using iterator = move_orderer_iterator;

  move_orderer_data data_;

  move_orderer_iterator begin() { return move_orderer_iterator(data_, 0); }
  move_orderer_iterator end() { return move_orderer_iterator(data_.list->size()); }

  move_orderer& set_first(const move& mv) {
    data_.set_first(mv);
    return *this;
  }

  move_orderer(const move_orderer_data& data) : data_{data} {}
};

}  // namespace chess
