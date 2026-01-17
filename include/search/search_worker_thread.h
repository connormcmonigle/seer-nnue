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

#include <search/search_worker.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

namespace search {

struct search_worker_thread {
  search_worker_external_state external_state_;

  std::atomic_bool should_exit_{false};
  std::atomic_bool is_searching_{false};
  std::atomic_bool is_initialized_{false};

  std::mutex thread_to_orchestrator_mutex_{};
  std::condition_variable thread_to_orchestrator_cv_{};

  std::mutex orchestrator_to_thread_to_mutex_{};
  std::condition_variable orchestrator_to_thread_cv_{};

  std::unique_ptr<search_worker> worker_{nullptr};
  std::unique_ptr<std::thread> worker_thread_{nullptr};

  search_worker_thread(const search_worker_external_state& external_state) noexcept : external_state_{external_state} {
    worker_thread_ = std::make_unique<std::thread>([this] { run_loop_(); });

    std::unique_lock lock(thread_to_orchestrator_mutex_);
    thread_to_orchestrator_cv_.wait(lock, [this] { return is_initialized_.load(std::memory_order_seq_cst); });
  }

  void go(const chess::board_history& hist, const chess::board& bd, const depth_type& start_depth) noexcept {}
  void stop() noexcept {}

  void run_loop_() noexcept {
    {
      std::unique_lock lock(thread_to_orchestrator_mutex_);
      worker_ = std::make_unique<search_worker>(external_state_);

      is_initialized_.store(true, std::memory_order_seq_cst);
      thread_to_orchestrator_cv_.notify_one();
    }

    while (!should_exit_.load(std::memory_order_seq_cst)) {
        {
            std::unique_lock lock(orchestrator_to_thread_to_mutex_);
            orchestrator_to_thread_cv_.wait(lock, [this] { return is_searching_.load(std::memory_order_seq_cst); });
        }

        worker_->iterative_deepening_loop();

        {
            
        }
    }
  }
};

}  // namespace search