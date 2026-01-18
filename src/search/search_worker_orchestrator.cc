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
#include <numeric>

namespace search {

void worker_orchestrator::reset() noexcept {
  tt_->clear();
  for (auto& worker_thread : worker_threads_) { worker_thread->worker().internal.reset(); };
}

void worker_orchestrator::resize(const std::size_t& new_size) noexcept {
  constants_->update_(new_size);
  
  const std::size_t old_size = worker_threads_.size();
  worker_threads_.resize(new_size);

  for (std::size_t i(old_size); i < new_size; ++i) {
    const search_worker_external_state external_state{weights_, tt_, constants_};
    worker_threads_[i] = std::make_unique<search_worker_thread>(external_state);
  }
}

void worker_orchestrator::go(const chess::board_history& hist, const chess::board& bd) noexcept {
  std::lock_guard access_lock(access_mutex_);

  tt_->update_gen();
  for (std::size_t i(0); i < worker_threads_.size(); ++i) {
    const depth_type start_depth = 1 + static_cast<depth_type>(i % 2);
    worker_threads_[i]->go(hist, bd, start_depth);
  }

  is_searching_.store(true);
}

void worker_orchestrator::stop() noexcept {
  std::lock_guard access_lock(access_mutex_);
  std::for_each(worker_threads_.begin(), worker_threads_.end(), [](auto& worker_thread) { worker_thread->stop(); });
  is_searching_.store(false);
}

bool worker_orchestrator::is_searching() noexcept {
  std::lock_guard access_lock(access_mutex_);
  return is_searching_.load();
}

std::size_t worker_orchestrator::nodes() const noexcept {
  return std::accumulate(worker_threads_.begin(), worker_threads_.end(), static_cast<std::size_t>(0), [](const std::size_t& count, const auto& worker_thread) {
    return count + worker_thread->worker().nodes();
  });
}

std::size_t worker_orchestrator::tb_hits() const noexcept {
  return std::accumulate(worker_threads_.begin(), worker_threads_.end(), static_cast<std::size_t>(0), [](const std::size_t& count, const auto& worker_thread) {
    return count + worker_thread->worker().tb_hits();
  });
}

search_worker& worker_orchestrator::primary_worker() noexcept { return worker_threads_[primary_id]->worker(); }

worker_orchestrator::worker_orchestrator(
    const nnue::quantized_weights* weights,
    const std::size_t hash_table_size,
    std::function<void(const search_worker&)> on_iter,
    std::function<void(const search_worker&)> on_update) noexcept {
  weights_ = weights;
  tt_ = std::make_shared<transposition_table>(hash_table_size);
  constants_ = std::make_shared<search_constants>();
  
  const search_worker_external_state external_state{weights, tt_, constants_, on_iter, on_update};
  worker_threads_.push_back(std::make_unique<search_worker_thread>(external_state));
}

}  // namespace search
