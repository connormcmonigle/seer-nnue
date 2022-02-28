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

  square_set threatened{};

  const std::unordered_map<move, std::uint64_t>* node_distribution{nullptr};
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

  move_orderer_data& set_threatened(const square_set& mask) {
    threatened = mask;
    return *this;
  }

  move_orderer_data& set_node_distribution(const std::unordered_map<move, std::uint64_t>* distribution) {
    node_distribution = distribution;
    return *this;
  }

  move_orderer_data(const board* bd_, const move_list* list_, const history_heuristic* hh_) : bd{bd_}, list{list_}, hh{hh_} {}
};

struct move_orderer_entry {
  using value_ = bit::range<std::uint32_t, 0, 32>;
  using killer_ = bit::next_flag<value_>;
  using positive_noisy_ = bit::next_flag<killer_>;
  using first_ = bit::next_flag<positive_noisy_>;

  move mv;
  std::uint64_t data_;

  const std::uint64_t& sort_key() const { return data_; }

  move_orderer_entry() = default;

  move_orderer_entry(const move& mv_, const std::uint64_t& data) : mv{mv_}, data_{data} {}

  move_orderer_entry(const move& mv_, bool is_first, bool is_positive_noisy, bool is_killer, std::int32_t value) : mv{mv_}, data_{0} {
    first_::set(data_, is_first);
    positive_noisy_::set(data_, is_positive_noisy);
    killer_::set(data_, is_killer);
    value_::set(data_, make_positive(value));
  }

  static inline move_orderer_entry make_first(const move& mv) { return move_orderer_entry(mv, true, false, false, 0); }

  static inline move_orderer_entry make_noisy(const move& mv, const std::int32_t& see_value, const std::int32_t& history_value) {
    const bool positive_noisy = see_value > 0;
    return move_orderer_entry(mv, false, positive_noisy, false, positive_noisy ? see_value : history_value);
  }

  static inline move_orderer_entry make_quiet(const move& mv, const move& killer, const std::int32_t& history_value) {
    return move_orderer_entry(mv, false, false, mv == killer, history_value);
  }

  static inline move_orderer_entry make_root(const move& mv, const std::uint64_t& node_count) { return move_orderer_entry(mv, node_count); }
};

template <bool is_root>
struct move_orderer_iterator_end_tag {};

template <bool is_root>
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
    auto comparator = [](const move_orderer_entry& a, const move_orderer_entry& b) { return a.sort_key() < b.sort_key(); };
    std::iter_swap(begin_, std::max_element(begin_, end_, comparator));
  }

  move_orderer_iterator& operator++() {
    ++begin_;
    if (begin_ != end_) { update_list_(); }
    return *this;
  }

  move_orderer_iterator operator++(int) {
    auto retval = *this;
    ++(*this);
    return retval;
  }

  constexpr bool operator==(const move_orderer_iterator<is_root>&) const { return false; }
  constexpr bool operator==(const move_orderer_iterator_end_tag<is_root>&) const { return begin_ == end_; }

  template <typename T>
  constexpr bool operator!=(const T& other) const {
    return !(*this == other);
  }

  std::tuple<std::ptrdiff_t, move> operator*() const { return std::tuple(begin_ - entries_.begin(), begin_->mv); }

  move_orderer_iterator(const move_orderer_data& data) : entries_{}, begin_{entries_.begin()} {
    if (is_root && data.node_distribution != nullptr) {
      end_ = std::transform(data.list->begin(), data.list->end(), entries_.begin(), [&data](const move& mv) {
        const auto iter = data.node_distribution->find(mv);
        return move_orderer_entry::make_root(mv, iter != data.node_distribution->end() ? iter->second : std::uint64_t{});
      });
    } else {
      const history::context ctxt{data.follow, data.counter, data.threatened};
      end_ = std::transform(data.list->begin(), data.list->end(), entries_.begin(), [&data, &ctxt](const move& mv) {
        if (mv == data.first) { return move_orderer_entry::make_first(mv); }
        if (mv.is_noisy()) { return move_orderer_entry::make_noisy(mv, data.bd->see<std::int32_t>(mv), data.hh->compute_value(ctxt, mv)); }
        return move_orderer_entry::make_quiet(mv, data.killer, data.hh->compute_value(ctxt, mv));
      });
    }

    update_list_();
  }
};

template <bool is_root>
struct move_orderer {
  using iterator = move_orderer_iterator<is_root>;

  move_orderer_data data_;

  move_orderer_iterator<is_root> begin() { return move_orderer_iterator<is_root>(data_); }
  move_orderer_iterator_end_tag<is_root> end() { return move_orderer_iterator_end_tag<is_root>(); }

  move_orderer<is_root>& set_first(const move& mv) {
    data_.set_first(mv);
    return *this;
  }

  move_orderer<is_root>& set_node_distribution(const std::unordered_map<move, std::uint64_t>* distribution) {
    data_.set_node_distribution(distribution);
    return *this;
  }

  move_orderer(const move_orderer_data& data) : data_{data} {}
};

}  // namespace chess
