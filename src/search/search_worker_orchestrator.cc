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

#include <search/search_worker_orchestrator.h>

#include <algorithm>

namespace search {

void worker_orchestrator::reset() noexcept {
  tt_->clear();
  for (auto& worker : workers_) { worker->internal.reset(); };
}

void worker_orchestrator::resize(const std::size_t& new_size) noexcept {
  constants_->update_(new_size);
  const std::size_t old_size = workers_.size();
  workers_.resize(new_size);
  for (std::size_t i(old_size); i < new_size; ++i) { workers_[i] = std::make_unique<search_worker>(weights_, tt_, constants_); }
}

void worker_orchestrator::go(const chess::board_history& hist, const chess::board& bd) noexcept {
  std::lock_guard access_lock(access_mutex_);
  std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
  std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
  threads_.clear();

  tt_->update_gen();
  for (std::size_t i(0); i < workers_.size(); ++i) {
    const depth_type start_depth = 1 + static_cast<depth_type>(i % 2);
    workers_[i]->go(hist, bd, start_depth);
  }

  std::transform(workers_.begin(), workers_.end(), std::back_inserter(threads_), [](auto& worker) {
    return std::thread([&worker] { worker->iterative_deepening_loop(); });
  });

  is_searching_.store(true);
}

void worker_orchestrator::stop() noexcept {
  std::lock_guard access_lock(access_mutex_);
  std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
  is_searching_.store(false);
}

bool worker_orchestrator::is_searching() noexcept {
  std::lock_guard access_lock(access_mutex_);
  return is_searching_.load();
}

std::size_t worker_orchestrator::nodes() const noexcept {
  return std::accumulate(
      workers_.begin(), workers_.end(), static_cast<std::size_t>(0), [](const std::size_t& count, const auto& worker) { return count + worker->nodes(); });
}

std::size_t worker_orchestrator::tb_hits() const noexcept {
  return std::accumulate(
      workers_.begin(), workers_.end(), static_cast<std::size_t>(0), [](const std::size_t& count, const auto& worker) { return count + worker->tb_hits(); });
}

search_worker& worker_orchestrator::primary_worker() noexcept { return *workers_[primary_id]; }

worker_orchestrator::worker_orchestrator(
    const nnue::weights* weights,
    std::size_t hash_table_size,
    std::function<void(const search_worker&)> on_iter,
    std::function<void(const search_worker&)> on_update) noexcept {
  weights_ = weights;
  tt_ = std::make_shared<transposition_table>(hash_table_size);
  constants_ = std::make_shared<search_constants>();
  workers_.push_back(std::make_unique<search_worker>(weights, tt_, constants_, on_iter, on_update));
}

worker_orchestrator::~worker_orchestrator() noexcept {
  std::lock_guard access_lock(access_mutex_);
  std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
  std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
}

}  // namespace search