#pragma once

#include <move.h>

#include <array>

namespace chess {

struct counter_move_heuristic {
  static constexpr size_t num_squares = 64;
  std::array<move, num_squares * num_squares> data{};

  size_t idx_(const move& mv) const {
    const size_t from = static_cast<size_t>(mv.from().index());
    const size_t to = static_cast<size_t>(mv.to().index());
    return from * num_squares + to;
  }

  counter_move_heuristic& update(const move& counter, const move& mv) {
    data[idx_(counter)] = mv;
    return *this;
  }

  move counter(const move& counter) const { return data[idx_(counter)]; }

  counter_move_heuristic& clear() {
    data.fill(move::null());
    return *this;
  }

  counter_move_heuristic() { clear(); }
};

struct sided_counter_move_heuristic : sided<sided_counter_move_heuristic, counter_move_heuristic> {
  counter_move_heuristic white;
  counter_move_heuristic black;

  sided_counter_move_heuristic& clear() {
    white.clear();
    black.clear();
    return *this;
  }

  sided_counter_move_heuristic() : white{}, black{} {}
};

}  // namespace chess