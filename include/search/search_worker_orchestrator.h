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

#include <search/search_worker.h>
#include <search/transposition_table.h>

#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace search {

struct worker_orchestrator {
  static constexpr std::size_t primary_id = 0;

  const nnue::quantized_weights* weights_;
  std::shared_ptr<transposition_table> tt_{nullptr};
  std::shared_ptr<search_constants> constants_{nullptr};

  std::mutex access_mutex_{};
  std::atomic_bool is_searching_{};
  std::vector<std::unique_ptr<search_worker>> workers_{};
  std::vector<std::thread> threads_{};

  void reset() noexcept;
  void resize(const std::size_t& new_size) noexcept;

  void go(const chess::board_history& hist, const chess::board& bd) noexcept;
  void stop() noexcept;

  [[nodiscard]] bool is_searching() noexcept;

  [[nodiscard]] std::size_t nodes() const noexcept;
  [[nodiscard]] std::size_t tb_hits() const noexcept;

  [[nodiscard]] search_worker& primary_worker() noexcept;

  worker_orchestrator(
      const nnue::quantized_weights* weights,
      std::size_t hash_table_size,
      std::function<void(const search_worker&)> on_iter = [](auto&&...) {},
      std::function<void(const search_worker&)> on_update = [](auto&&...) {}) noexcept;

  ~worker_orchestrator() noexcept;
};

}  // namespace search