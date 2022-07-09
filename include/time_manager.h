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
#include <search_constants.h>

#include <atomic>
#include <functional>
#include <mutex>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>

namespace engine {

namespace go {

struct wtime {
  static constexpr std::string_view name = "wtime";
  using type = int;
};

struct btime {
  static constexpr std::string_view name = "btime";
  using type = int;
};

struct winc {
  static constexpr std::string_view name = "winc";
  using type = int;
};

struct binc {
  static constexpr std::string_view name = "binc";
  using type = int;
};

struct movetime {
  static constexpr std::string_view name = "movetime";
  using type = int;
};

struct movestogo {
  static constexpr std::string_view name = "movestogo";
  using type = int;
};

struct depth {
  static constexpr std::string_view name = "depth";
  using type = search::depth_type;
};

struct nodes {
  static constexpr std::string_view name = "nodes";
  using type = size_t;
};

struct ponder {
  static constexpr std::string_view name = "ponder";
};

struct infinite {
  static constexpr std::string_view name = "infinite";
};

}  // namespace go

template <typename N>
struct named_numeric_param {
  static constexpr std::string_view name_ = N::name;
  std::optional<typename N::type> data_{std::nullopt};

  constexpr std::string_view name() const { return name_; }
  std::optional<typename N::type> data() const { return data_; }
  void set_data(const typename N::type& value) { data_ = value; }
  void clear_data() { data_ = std::nullopt; }

  void read(const std::string& line) {
    std::regex pattern(".*" + std::string(name_) + " ([-+]?[0-9]+).*");
    std::smatch matches{};
    if (std::regex_search(line, matches, pattern)) {
      typename N::type val{};
      std::stringstream(matches.str(1)) >> val;
      data_ = val;
    } else {
      data_ = std::nullopt;
    }
  }
};

template <typename N>
struct named_condition {
  static constexpr std::string_view name_ = N::name;
  bool data_{false};

  constexpr std::string_view name() const { return name_; }
  bool data() const { return data_; }
  void set_data(const bool& value) { data_ = value; }

  void read(const std::string& line) {
    std::regex pattern(".*" + std::string(name_) + ".*");
    std::smatch matches{};
    data_ = (std::regex_search(line, matches, pattern));
  }
};

struct update_info {
  size_t nodes;
};

struct iter_info {
  search::depth_type depth;
  size_t best_move_percent;
};

struct time_manager {
  std::mutex access_mutex_;

  std::tuple<
      named_numeric_param<go::wtime>,
      named_numeric_param<go::btime>,
      named_numeric_param<go::winc>,
      named_numeric_param<go::binc>,
      named_numeric_param<go::movetime>,
      named_numeric_param<go::movestogo>,
      named_numeric_param<go::depth>,
      named_numeric_param<go::nodes>,
      named_condition<go::ponder>,
      named_condition<go::infinite> >
      params_{};

  std::chrono::steady_clock::time_point search_start{};
  std::optional<std::chrono::milliseconds> min_budget{};
  std::optional<std::chrono::milliseconds> max_budget{};

  std::chrono::milliseconds elapsed() const {
    return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - search_start);
  }

  template <typename N, size_t idx = 0>
  auto& get() {
    constexpr std::string_view name_idx = std::tuple_element<idx, decltype(params_)>::type::name_;
    if constexpr (name_idx == N::name) {
      return std::get<idx>(params_);
    } else {
      return get<N, idx + 1>();
    }
  }

  bool is_pondering() { return get<go::ponder>().data(); }

  void ponder_hit() {
    std::lock_guard<std::mutex> access_lk(access_mutex_);
    search_start = std::chrono::steady_clock::now();
    get<go::ponder>().set_data(false);
  }

  std::chrono::milliseconds our_time(const bool& pov) {
    const auto x = pov ? get<go::wtime>().data() : get<go::btime>().data();
    return std::chrono::milliseconds(x.value_or(10000));
  }

  std::chrono::milliseconds our_increment(const bool& pov) {
    const auto x = pov ? get<go::winc>().data() : get<go::binc>().data();
    return std::chrono::milliseconds(x.value_or(0));
  }

  time_manager& read(const std::string& line) {
    util::apply(params_, [&line](auto& param) { param.read(line); });
    return *this;
  }

  time_manager& init(const bool& pov, const std::string& line) {
    constexpr auto over_head = std::chrono::milliseconds(5);

    std::lock_guard<std::mutex> access_lk(access_mutex_);
    search_start = std::chrono::steady_clock::now();
    read(line);

    if (get<go::depth>().data().has_value() || get<go::infinite>().data()) {
      min_budget = std::nullopt;
      max_budget = std::nullopt;
    } else if (get<go::movetime>().data().has_value()) {
      min_budget = std::nullopt;
      max_budget = std::chrono::milliseconds(*get<go::movetime>().data());
    } else {
      const auto remaining = our_time(pov);
      const auto inc = our_increment(pov);

      if (get<go::movestogo>().data().has_value()) {
        // handle cyclical time controls (x / y + z)
        const go::movestogo::type moves_to_go = *get<go::movestogo>().data();
        min_budget = 2 * (remaining - over_head) / (3 * moves_to_go) + inc;
        max_budget = 10 * (remaining - over_head) / (3 * moves_to_go) + inc;
      } else {
        // handle incremental time controls (x + z)
        min_budget = (remaining - over_head + 25 * inc) / 25;
        max_budget = (remaining - over_head + 25 * inc) / 5;
      }

      // avoid time losses by capping budget to 4/5 remaining time
      min_budget = std::min(4 * (remaining - over_head) / 5, *min_budget);
      max_budget = std::min(4 * (remaining - over_head) / 5, *max_budget);
    }

    return *this;
  }

  bool should_stop_on_update(const update_info& info) {
    std::lock_guard<std::mutex> access_lk(access_mutex_);
    if (get<go::infinite>().data()) { return false; }
    if (get<go::ponder>().data()) { return false; }
    if (get<go::nodes>().data().has_value() && info.nodes >= *get<go::nodes>().data()) { return true; }
    if (max_budget.has_value() && elapsed() >= *max_budget) { return true; };
    return false;
  }

  bool should_stop_on_iter(const iter_info& info) {
    constexpr size_t numerator = 50;
    constexpr size_t min_percent = 20;
    
    std::lock_guard<std::mutex> access_lk(access_mutex_);
    if (get<go::infinite>().data()) { return false; }
    if (get<go::ponder>().data()) { return false; }

    if (info.depth >= search::max_depth) { return true; }
    if (max_budget.has_value() && elapsed() >= *max_budget) { return true; }
    if (min_budget.has_value() && elapsed() >= (*min_budget * numerator / std::max(info.best_move_percent, min_percent))) { return true; }
    if (get<go::depth>().data().has_value() && info.depth >= *get<go::depth>().data()) { return true; }
    return false;
  }
};

template <typename T>
struct simple_timer {
  std::mutex start_mutex_;
  std::chrono::steady_clock::time_point start_;

  T elapsed() {
    std::lock_guard<std::mutex> start_lk(start_mutex_);
    return std::chrono::duration_cast<T>(std::chrono::steady_clock::now() - start_);
  }

  simple_timer<T>& lap() {
    std::lock_guard<std::mutex> start_lk(start_mutex_);
    start_ = std::chrono::steady_clock::now();
    return *this;
  }

  simple_timer() : start_{std::chrono::steady_clock::now()} {}
};

}  // namespace engine
