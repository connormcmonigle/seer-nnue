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

#include <move.h>
#include <zobrist_util.h>

#include <vector>

namespace chess {

template <typename T>
struct popper {
  T* data_;
  popper(T* data) : data_{data} {}
  ~popper() { data_->pop_(); }
};

template <typename T, typename U>
struct base_history {
  using stored_type = U;
  std::vector<stored_type> history_;

  T& cast() { return static_cast<T&>(*this); }
  const T& cast() const { return static_cast<const T&>(*this); }

  T& clear() {
    history_.clear();
    return cast();
  }

  popper<T> scoped_push_(const stored_type& elem) {
    history_.push_back(elem);
    return popper<T>(&cast());
  }

  T& push_(const stored_type& elem) {
    history_.push_back(elem);
    return cast();
  }

  T& pop_() {
    history_.pop_back();
    return cast();
  }

  stored_type back() const { return history_.back(); }

  size_t len() const { return history_.size(); }

  base_history() : history_{} {}
  base_history(const std::vector<stored_type>& h) : history_{h} {}
};

struct position_history : base_history<position_history, zobrist::hash_type> {
  size_t occurrences(const zobrist::hash_type& hash) const {
    size_t occurrences_{0};
    for (auto it = history_.crbegin(); it != history_.crend(); ++it) { occurrences_ += static_cast<size_t>(*it == hash); }
    return occurrences_;
  }

  bool is_two_fold(const zobrist::hash_type& hash) const { return occurrences(hash) >= 1; }
};

}  // namespace chess
