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

#include <search_worker.h>

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace search {

struct worker_orchestrator {
  using worker_type = search_worker<>;
  static constexpr size_t primary_id = 0;

  const nnue::weights* weights_;
  std::shared_ptr<transposition_table> tt_{nullptr};
  std::shared_ptr<search_constants> constants_{nullptr};

  std::mutex access_mutex_{};
  std::atomic_bool is_searching_{};
  std::vector<std::unique_ptr<worker_type>> workers_{};
  std::vector<std::thread> threads_{};

  void reset() {
    tt_->clear();
    for (auto& worker : workers_) { worker->internal.reset(); };
  }

  void resize(const size_t& new_size) {
    constants_->update_(new_size);
    const size_t old_size = workers_.size();
    workers_.resize(new_size);
    for (size_t i(old_size); i < new_size; ++i) { workers_[i] = std::make_unique<worker_type>(weights_, tt_, constants_); }
  }

  void go(const chess::position_history& hist, const chess::board& bd) {
    std::lock_guard access_lock(access_mutex_);
    std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
    std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
    threads_.clear();

    tt_->update_gen();
    for (size_t i(0); i < workers_.size(); ++i) {
      const depth_type start_depth = 1 + static_cast<depth_type>(i % 2);
      workers_[i]->go(hist, bd, start_depth);
    }

    std::transform(workers_.begin(), workers_.end(), std::back_inserter(threads_), [](auto& worker) {
      return std::thread([&worker] { worker->iterative_deepening_loop(); });
    });

    is_searching_.store(true);
  }

  void stop() {
    std::lock_guard access_lock(access_mutex_);
    std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
    is_searching_.store(false);
  }

  bool is_searching() {
    std::lock_guard access_lock(access_mutex_);
    return is_searching_.load();
  }

  size_t nodes() const {
    return std::accumulate(
        workers_.begin(), workers_.end(), static_cast<size_t>(0), [](const size_t& count, const auto& worker) { return count + worker->nodes(); });
  }

  size_t tb_hits() const {
    return std::accumulate(
        workers_.begin(), workers_.end(), static_cast<size_t>(0), [](const size_t& count, const auto& worker) { return count + worker->tb_hits(); });
  }

  worker_type& primary_worker() { return *workers_[primary_id]; }

  worker_orchestrator(
      const nnue::weights* weights,
      size_t hash_table_size,
      std::function<void(const worker_type&)> on_iter = [](auto&&...) {},
      std::function<void(const worker_type&)> on_update = [](auto&&...) {}) {
    weights_ = weights;
    tt_ = std::make_shared<transposition_table>(hash_table_size);
    constants_ = std::make_shared<search_constants>();
    workers_.push_back(std::make_unique<worker_type>(weights, tt_, constants_, on_iter, on_update));
  }

  ~worker_orchestrator() {
    std::lock_guard access_lock(access_mutex_);
    std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
    std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
  }
};

}  // namespace search