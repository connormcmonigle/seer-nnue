/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

#include <chess/board.h>
#include <chess/move.h>
#include <chess/move_list.h>
#include <search/history_heuristic.h>
#include <util/bit_range.h>

#include <cstdint>
#include <iterator>
#include <limits>
#include <optional>
#include <tuple>

namespace search {

[[nodiscard]] constexpr std::uint32_t make_positive(const std::int32_t& x) noexcept {
  constexpr std::uint32_t upper = static_cast<std::uint32_t>(1) + std::numeric_limits<std::int32_t>::max();
  return upper + x;
}

struct move_orderer_data {
  chess::move killer{chess::move::null()};
  chess::move follow{chess::move::null()};
  chess::move counter{chess::move::null()};
  chess::move previous_follow{chess::move::null()};
  chess::move previous_counter{chess::move::null()};
  chess::move first{chess::move::null()};

  chess::square_set threatened{};
  zobrist::hash_type pawn_hash{};

  const chess::board* bd;
  const history_heuristic* hh;

  [[maybe_unused]] constexpr move_orderer_data& set_killer(const chess::move& mv) noexcept {
    killer = mv;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_follow(const chess::move& mv) noexcept {
    follow = mv;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_counter(const chess::move& mv) noexcept {
    counter = mv;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_previous_follow(const chess::move& mv) noexcept {
    previous_follow = mv;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_previous_counter(const chess::move& mv) noexcept {
    previous_counter = mv;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_first(const chess::move& mv) noexcept {
    first = mv;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_threatened(const chess::square_set& mask) noexcept {
    threatened = mask;
    return *this;
  }

  [[maybe_unused]] constexpr move_orderer_data& set_pawn_hash(const zobrist::hash_type& hash) noexcept {
    pawn_hash = hash;
    return *this;
  }

  constexpr move_orderer_data(const chess::board* bd_, const history_heuristic* hh_) noexcept : bd{bd_}, hh{hh_} {}
};

struct move_orderer_entry {
  using value_ = util::bit_range<std::uint32_t, 0, 32>;
  using killer_ = util::next_bit_flag<value_>;
  using positive_noisy_ = util::next_bit_flag<killer_>;
  using first_ = util::next_bit_flag<positive_noisy_>;

  chess::move mv;
  std::uint64_t data_;

  const std::uint64_t& sort_key() const { return data_; }

  move_orderer_entry() = default;

  constexpr move_orderer_entry(const chess::move& mv_, bool is_positive_noisy, bool is_killer, std::int32_t value) noexcept : mv{mv_}, data_{0} {
    positive_noisy_::set(data_, is_positive_noisy);
    killer_::set(data_, is_killer);
    value_::set(data_, make_positive(value));
  }

  [[nodiscard]] static constexpr move_orderer_entry make_noisy(const chess::move& mv, const bool positive_noisy, const std::int32_t& history_value) {
    return move_orderer_entry(mv, positive_noisy, false, positive_noisy ? mv.mvv_lva_key<std::int32_t>() : history_value);
  }

  [[nodiscard]] static constexpr move_orderer_entry make_quiet(const chess::move& mv, const chess::move& killer, const std::int32_t& history_value) {
    return move_orderer_entry(mv, false, mv == killer, history_value);
  }
};

struct move_orderer_stepper {
  using entry_array_type = std::array<move_orderer_entry, chess::move_list::max_branching_factor>;

  bool is_initialized_{false};

  entry_array_type entries_;
  entry_array_type::iterator begin_;
  entry_array_type::iterator end_;

  [[nodiscard]] constexpr bool is_initialized() const noexcept { return is_initialized_; }
  [[nodiscard]] constexpr bool has_next() const noexcept { return begin_ != end_; }
  [[nodiscard]] constexpr chess::move current_move() const noexcept { return begin_->mv; }

  [[maybe_unused]] move_orderer_stepper& initialize(const move_orderer_data& data, const chess::move_list& list) noexcept;
  inline void update_list_() const noexcept;
  void next() noexcept;

  move_orderer_stepper() noexcept : begin_{entries_.begin()} {}

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
  using iterator_category = std::input_iterator_tag;

  int idx{};
  move_orderer_stepper stepper_;
  move_orderer_data data_;

  [[nodiscard]] std::tuple<int, chess::move> operator*() const noexcept;
  [[maybe_unused]] move_orderer_iterator<mode>& operator++() noexcept;

  [[nodiscard]] constexpr bool operator==(const move_orderer_iterator<mode>&) const noexcept { return false; }

  [[nodiscard]] constexpr bool operator==(const move_orderer_iterator_end_tag&) const noexcept {
    return stepper_.is_initialized() && !stepper_.has_next();
  }

  template <typename T>
  [[nodiscard]] constexpr bool operator!=(const T& other) const noexcept {
    return !(*this == other);
  }

  move_orderer_iterator(const move_orderer_data& data) noexcept;
};

template <typename mode>
struct move_orderer {
  using iterator = move_orderer_iterator<mode>;

  move_orderer_data data_;

  [[nodiscard]] move_orderer_iterator<mode> begin() const noexcept { return move_orderer_iterator<mode>(data_); }
  [[nodiscard]] move_orderer_iterator_end_tag end() const noexcept { return move_orderer_iterator_end_tag(); }

  [[maybe_unused]] move_orderer& set_first(const chess::move& mv) noexcept {
    data_.set_first(mv);
    return *this;
  }

  move_orderer(const move_orderer_data& data) noexcept : data_{data} {}
};

}  // namespace search
