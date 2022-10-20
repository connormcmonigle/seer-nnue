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
#include <condition_variable>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>

namespace search {

enum class worker_orchestrator_loop_state {
  waiting,
  starting,
  searching,
  stopping,
  terminating,
};

struct worker_orchestrator_loop_data {
  worker_orchestrator_loop_state state{worker_orchestrator_loop_state::waiting};
  std::function<void()> on_stopped{[] {}};
  std::mutex cv_mutex{};
  std::condition_variable cv{};
};

struct worker_orchestrator_shared_state {
  const nnue::weights* weights;
  std::shared_ptr<transposition_table> tt{nullptr};
  std::shared_ptr<search_constants> constants{nullptr};
};

struct worker_orchestrator {
  static constexpr size_t primary_id = 0;

  chess::position_history history_{};
  chess::board board_{chess::board::start_pos()};

  worker_orchestrator_shared_state shared_state_;
  std::vector<std::unique_ptr<search_worker>> workers_{};
  std::vector<std::thread> threads_{};

  worker_orchestrator_loop_data loop_data_;
  std::thread loop_thread_{};

  void reset() {
    shared_state_.tt->clear();
    for (auto& worker : workers_) { worker->internal.reset(); };
  }

  void resize(const size_t& new_size) {
    shared_state_.constants->update_(new_size);
    const size_t old_size = workers_.size();
    workers_.resize(new_size);
    for (size_t i(old_size); i < new_size; ++i) {
      workers_[i] = std::make_unique<search_worker>(shared_state_.weights, shared_state_.tt, shared_state_.constants);
    }
  }

  void go(const chess::position_history& hist, const chess::board& bd) {
    std::lock_guard cv_lock(loop_data_.cv_mutex);
    history_ = hist;
    board_ = bd;
    loop_data_.state = worker_orchestrator_loop_state::starting;
    loop_data_.cv.notify_one();
  }

  void stop(std::function<void()> on_stopped = [] {}) {
    std::lock_guard cv_lock(loop_data_.cv_mutex);
    loop_data_.on_stopped = on_stopped;
    workers_[primary_id]->stop();
    loop_data_.state = worker_orchestrator_loop_state::stopping;
    loop_data_.cv.notify_one();
  }

  void loop_() {
    for (bool terminated{false}; !terminated;) {
      std::unique_lock cv_lock(loop_data_.cv_mutex);
      loop_data_.cv.wait(cv_lock, [this] {
        switch (loop_data_.state) {
          case worker_orchestrator_loop_state::starting: return true;
          case worker_orchestrator_loop_state::stopping: return true;
          case worker_orchestrator_loop_state::terminating: return true;
          default: return false;
        }
      });

      switch (loop_data_.state) {
        case worker_orchestrator_loop_state::starting: {
          std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
          std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
          threads_.clear();

          std::for_each(workers_.begin(), workers_.end(), [this](auto& worker) { worker->go(history_, board_); });
          std::transform(workers_.begin(), workers_.end(), std::back_inserter(threads_), [](auto& worker) {
            return std::thread([&worker]() { worker->iterative_deepening_loop(); });
          });

          loop_data_.state = worker_orchestrator_loop_state::searching;
          break;
        }

        case worker_orchestrator_loop_state::stopping: {
          std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
          std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
          threads_.clear();
          loop_data_.on_stopped();
          loop_data_.on_stopped = [] {};

          loop_data_.state = worker_orchestrator_loop_state::waiting;
          break;
        }

        case worker_orchestrator_loop_state::terminating: {
          terminated = true;
          break;
        }

        default: break;
      }
    }

    std::for_each(workers_.begin(), workers_.end(), [](auto& worker) { worker->stop(); });
    std::for_each(threads_.begin(), threads_.end(), [](auto& thread) { thread.join(); });
    threads_.clear();
  }

  bool is_searching() {
    std::lock_guard cv_lock(loop_data_.cv_mutex);
    return loop_data_.state == worker_orchestrator_loop_state::searching;
  }

  size_t nodes() const {
    return std::accumulate(
        workers_.begin(), workers_.end(), static_cast<size_t>(0), [](const size_t& count, const auto& worker) { return count + worker->nodes(); });
  }

  size_t tb_hits() const {
    return std::accumulate(
        workers_.begin(), workers_.end(), static_cast<size_t>(0), [](const size_t& count, const auto& worker) { return count + worker->tb_hits(); });
  }

  search_worker& primary_worker() { return *workers_[primary_id]; }

  worker_orchestrator(
      const nnue::weights* weights,
      size_t hash_table_size,
      std::function<void(const search_worker&)> on_iter = [](auto&&...) {},
      std::function<void(const search_worker&)> on_update = [](auto&&...) {})
      : loop_thread_([this] { loop_(); }) {
    shared_state_.weights = weights;
    shared_state_.tt = std::make_shared<transposition_table>(hash_table_size);
    shared_state_.constants = std::make_shared<search_constants>();
    workers_.push_back(std::make_unique<search_worker>(weights, shared_state_.tt, shared_state_.constants, on_iter, on_update));
  }

  ~worker_orchestrator() {
    {
      std::lock_guard cv_lock(loop_data_.cv_mutex);
      loop_data_.state = worker_orchestrator_loop_state::terminating;
      loop_data_.cv.notify_one();
    }

    loop_thread_.join();
  }
};

}  // namespace search