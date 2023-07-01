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

#include <engine/time_manager.h>

#include <regex>

namespace engine {

time_manager& time_manager::ponder_hit() noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  search_start = std::chrono::steady_clock::now();
  ponder = false;

  return *this;
}

time_manager& time_manager::reset_() noexcept {
  search_start = std::chrono::steady_clock::now();

  min_budget = std::nullopt;
  max_budget = std::nullopt;

  depth_limit = std::nullopt;
  node_limit = std::nullopt;

  ponder = false;
  infinite = false;

  return *this;
}

time_manager& time_manager::init(const bool&, const go::infinite&) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  reset_();
  infinite = true;
  return *this;
}

time_manager& time_manager::init(const bool&, const go::depth& data) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  reset_();
  depth_limit = data.depth;

  return *this;
}

time_manager& time_manager::init(const bool&, const go::nodes& data) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  reset_();
  node_limit = data.nodes;

  return *this;
}

time_manager& time_manager::init(const bool&, const go::move_time& data) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  reset_();

  ponder = data.ponder;
  min_budget = std::nullopt;
  max_budget = data.move_time_ms();

  return *this;
}

time_manager& time_manager::init(const bool& pov, const go::increment& data) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  reset_();
  const auto remaining = data.our_time_ms(pov);
  const auto inc = data.our_increment_ms(pov);

  ponder = data.ponder;
  min_budget = (remaining - over_head + 25 * inc) / 25;
  max_budget = (remaining - over_head + 25 * inc) / 10;

  min_budget = std::min(4 * (remaining - over_head) / 5, *min_budget);
  max_budget = std::min(4 * (remaining - over_head) / 5, *max_budget);

  return *this;
}

time_manager& time_manager::init(const bool& pov, const go::moves_to_go& data) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  reset_();
  const auto remaining = data.our_time_ms(pov);

  ponder = data.ponder;
  min_budget = 2 * (remaining - over_head) / (3 * data.moves_to_go);
  max_budget = 10 * (remaining - over_head) / (3 * data.moves_to_go);

  return *this;
}

std::chrono::milliseconds time_manager::elapsed() const noexcept {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - search_start);
}

bool time_manager::should_stop_on_update(const update_info& info) noexcept {
  std::lock_guard<std::mutex> access_lk(access_mutex_);

  if (infinite) { return false; }
  if (ponder) { return false; }

  if (node_limit.has_value() && info.nodes >= *node_limit) { return true; }
  if (max_budget.has_value() && elapsed() >= *max_budget) { return true; };
  return false;
}

bool time_manager::should_stop_on_iter(const iter_info& info) noexcept {
  constexpr std::size_t numerator = 50;
  constexpr std::size_t min_percent = 20;

  std::lock_guard<std::mutex> access_lk(access_mutex_);

  if (infinite) { return false; }
  if (ponder) { return false; }

  if (info.depth >= search::max_depth) { return true; }
  if (max_budget.has_value() && elapsed() >= *max_budget) { return true; }
  if (min_budget.has_value() && elapsed() >= (*min_budget * numerator / std::max(info.best_move_percent, min_percent))) { return true; }
  if (depth_limit.has_value() && info.depth >= *depth_limit) { return true; }
  return false;
}

template <typename T>
T simple_timer<T>::elapsed() noexcept {
  std::lock_guard<std::mutex> start_lk(start_mutex_);
  return std::chrono::duration_cast<T>(std::chrono::steady_clock::now() - start_);
}

template <typename T>
simple_timer<T>& simple_timer<T>::lap() noexcept {
  std::lock_guard<std::mutex> start_lk(start_mutex_);
  start_ = std::chrono::steady_clock::now();
  return *this;
}

}  // namespace engine

template class engine::simple_timer<std::chrono::nanoseconds>;
template class engine::simple_timer<std::chrono::milliseconds>;
