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

#include <search/search_constants.h>

#include <cstddef>
#include <mutex>
#include <optional>

namespace engine {

namespace go {
struct infinite {};

struct depth {
  search::depth_type depth;
};

struct nodes {
  std::size_t nodes;
};

struct move_time {
  bool ponder;
  int move_time;

  [[nodiscard]] constexpr std::chrono::milliseconds move_time_ms() const noexcept {
    const int value = move_time;
    return std::chrono::milliseconds(value);
  }
};

struct timed_move_state {
  bool ponder;

  int white_time;
  int black_time;

  [[nodiscard]] constexpr std::chrono::milliseconds our_time_ms(const bool& pov) const noexcept {
    const int value = pov ? white_time : black_time;
    return std::chrono::milliseconds(value);
  }
};

struct increment : public timed_move_state {
  int white_increment;
  int black_increment;

  [[nodiscard]] constexpr std::chrono::milliseconds our_increment_ms(const bool& pov) const noexcept {
    const int value = pov ? white_increment : black_increment;
    return std::chrono::milliseconds(value);
  }
};

struct moves_to_go : public timed_move_state {
  int moves_to_go;
};

}  // namespace go

struct update_info {
  std::size_t nodes;
};

struct iter_info {
  search::depth_type depth;
  std::size_t best_move_percent;
};

struct time_manager {
  static constexpr auto over_head = std::chrono::milliseconds(50);

  std::mutex access_mutex_;

  std::chrono::steady_clock::time_point search_start{};
  std::optional<std::chrono::milliseconds> min_budget{};
  std::optional<std::chrono::milliseconds> max_budget{};

  std::optional<int> depth_limit{};
  std::optional<std::size_t> node_limit{};

  bool ponder{};
  bool infinite{};

  [[nodiscard]] constexpr bool is_pondering() const noexcept { return ponder; }
  [[maybe_unused]] time_manager& ponder_hit() noexcept;

  [[maybe_unused]] time_manager& reset_() noexcept;
  [[maybe_unused]] time_manager& init(const bool& pov, const go::infinite& data) noexcept;
  [[maybe_unused]] time_manager& init(const bool& pov, const go::depth& data) noexcept;
  [[maybe_unused]] time_manager& init(const bool& pov, const go::nodes& data) noexcept;

  [[maybe_unused]] time_manager& init(const bool& pov, const go::increment& data) noexcept;
  [[maybe_unused]] time_manager& init(const bool& pov, const go::move_time& data) noexcept;
  [[maybe_unused]] time_manager& init(const bool& pov, const go::moves_to_go& data) noexcept;

  [[nodiscard]] std::chrono::milliseconds elapsed() const noexcept;
  [[nodiscard]] bool should_stop_on_update(const update_info& info) noexcept;
  [[nodiscard]] bool should_stop_on_iter(const iter_info& info) noexcept;
};

template <typename T>
struct simple_timer {
  std::mutex start_mutex_;
  std::chrono::steady_clock::time_point start_;

  [[nodiscard]] T elapsed() noexcept;
  [[maybe_unused]] simple_timer<T>& lap() noexcept;

  simple_timer() noexcept : start_{std::chrono::steady_clock::now()} {}
};

}  // namespace engine
