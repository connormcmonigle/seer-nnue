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

#include <chess/move.h>
#include <search/search_worker.h>

#include <atomic>
#include <condition_variable>
#include <memory>
#include <thread>

namespace search {

enum class thread_state { initializing, pending, searching, done };

struct search_worker_thread {
  search_worker_external_state external_state_;
  std::atomic<thread_state> thread_state_{thread_state::initializing};

  std::mutex thread_to_caller_mutex_{};
  std::condition_variable thread_to_caller_cv_{};

  std::mutex caller_to_thread_to_mutex_{};
  std::condition_variable caller_to_thread_cv_{};

  std::unique_ptr<search_worker> worker_{nullptr};
  std::unique_ptr<std::thread> worker_thread_{nullptr};

  search_worker_thread(const search_worker_external_state& external_state) noexcept : external_state_{external_state} {
    worker_thread_ = std::make_unique<std::thread>([this] { thread_loop_(); });

    {
      std::unique_lock lock(thread_to_caller_mutex_);
      thread_to_caller_cv_.wait(lock, [this] { return thread_state_ == thread_state::pending; });
    }
  }

  [[nodiscard]] search_worker& worker() noexcept { return *worker_; }
  [[nodiscard]] const search_worker& worker() const noexcept { return *worker_; }

  void go(const chess::board_history& hist, const chess::board& bd, const depth_type& start_depth) noexcept {
    stop_sync_();
    worker_->go(hist, bd, start_depth);

    {
      std::unique_lock lock(caller_to_thread_to_mutex_);
      thread_state_ = thread_state::searching;
      caller_to_thread_cv_.notify_one();
    }
  }

  void stop() noexcept { stop_nosync_(); }

  void stop_nosync_() noexcept { worker_->stop(); }

  void stop_sync_() noexcept {
    worker_->stop();

    {
      std::unique_lock lock(thread_to_caller_mutex_);
      thread_to_caller_cv_.wait(lock, [this] { return thread_state_ == thread_state::pending; });
    }
  }

  void thread_loop_() noexcept {
    {
      std::unique_lock lock(thread_to_caller_mutex_);
      worker_ = std::make_unique<search_worker>(external_state_);
      thread_state_.store(thread_state::pending, std::memory_order_seq_cst);
      thread_to_caller_cv_.notify_one();
    }

    while (true) {
      {
        std::unique_lock lock(caller_to_thread_to_mutex_);
        caller_to_thread_cv_.wait(lock, [this] { return thread_state_ != thread_state::pending; });
      }

      if (thread_state_ == thread_state::done) { break; }
      if (thread_state_ == thread_state::searching) { worker_->iterative_deepening_loop(); }

      {
        std::unique_lock lock(thread_to_caller_mutex_);
        thread_state_ = thread_state::pending;
        thread_to_caller_cv_.notify_one();
      }
    }
  }

  ~search_worker_thread() {
    stop_sync_();

    {
      std::unique_lock lock(caller_to_thread_to_mutex_);
      thread_state_ = thread_state::done;
      caller_to_thread_cv_.notify_one();
    }

    worker_thread_->join();
  }
};

}  // namespace search
