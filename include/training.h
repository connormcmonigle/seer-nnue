#pragma once
#include <fstream>
#include <set>
#include <chrono>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <optional>


#include <enum_util.h>
#include <move.h>
#include <nnue_model.h>
#include <transposition_table.h>
#include <search_constants.h>
#include <time_manager.h>
#include <thread_worker.h>


namespace train{

using real_type = float;
using state_type = chess::board;
using score_type = search::score_type;
using wdl_type = search::wdl_type;

struct continuation_type{
  chess::position_history history_{};
  state_type state_{};
  
  const chess::position_history& history() const { return history_; }
  const state_type& state() const { return state_; }
};

constexpr size_t half_feature_numel_ = nnue::half_ka_numel;
constexpr size_t max_active_half_features_ = 32;
constexpr score_type wdl_scale = search::wdl_scale<score_type>;

constexpr auto win = wdl_type(wdl_scale, 0, 0);
constexpr auto draw = wdl_type(0, wdl_scale, 0);
constexpr auto loss = wdl_type(0, 0, wdl_scale);

struct feature_set : chess::sided<feature_set, std::set<size_t>>{
  std::set<size_t> white;
  std::set<size_t> black;

  feature_set() : white{}, black{} {}
};

namespace config{

constexpr size_t tt_mb_size = 1024;
constexpr search::depth_type init_depth = 1;
constexpr search::depth_type continuation_depth = 4;
constexpr search::depth_type continuation_max_length = 16;
constexpr std::chrono::seconds timeout = std::chrono::seconds(10);

}

std::tuple<bool, search::wdl_type> terminality(const chess::position_history& hist, const state_type& state){
  using return_type = std::tuple<bool, search::wdl_type>;

  if(hist.is_two_fold(state.hash())){ return return_type(true, draw); }
  if(state.generate_moves().size() == 0){ return return_type(true, state.is_check() ? loss : draw); }

  return return_type(false, search::wdl_type(0, 0, 0));
}

feature_set get_features(const state_type& state){
  feature_set features{};
  state.show_init(features);
  return features;
}

template<typename T>
struct train_interface{
  std::shared_ptr<search::constants> constants_ = std::make_shared<search::constants>(1);
  std::shared_ptr<chess::transposition_table> tt_ = std::make_shared<chess::transposition_table>(config::tt_mb_size);
  nnue::weights<T> weights_{};

  void load_weights(const std::string& path){ weights_.load(path); }

  search::wdl_type get_wdl(const continuation_type& continuation) const {
    if(const auto [is_terminal, wdl] = terminality(continuation.history(), continuation.state()); is_terminal){ return wdl; }
    auto evaluator = nnue::eval(&weights_);
    continuation.state().show_init(evaluator);
    return evaluator.wdl(continuation.state().turn());
  }

  std::optional<continuation_type> get_continuation(state_type state){
    using worker_type = chess::thread_worker<T, false>;

    engine::simple_timer<std::chrono::milliseconds> timer{};

    const size_t man_0 = state.num_pieces();
    auto hist = chess::position_history{};
    if(const auto terminal = terminality(hist, state); std::get<bool>(terminal)){ return continuation_type{hist, state}; }

    std::unique_ptr<worker_type> worker = std::make_unique<worker_type>(
      &weights_, tt_, constants_,
      [&timer, &worker](auto&&...){
        if(timer.elapsed() >= config::timeout){ worker.stop(); }
        if(worker.depth() >= config::continuation_depth){ worker.stop(); } 
      }
    );

    worker -> set_position(hist, state);
    worker -> go(config::init_depth);
    worker -> iterative_deepening_loop_();
    if(timer.elapsed() >= config::timeout){ return std::nullopt; }
    hist.push_(state.hash());

    if(!worker -> best_move().is_quiet()){ return std::nullopt; }

    chess::move last_move = worker.best_move();
    state = state.forward(worker.best_move());

    for(search::depth_type length{0}; length < config::continuation_max_length; ++length){
      if(const auto terminal = terminality(hist, state); std::get<bool>(terminal)){ return continuation_type{hist, state}; }
      if(last_move.is_quiet() && state.num_pieces() != man_0){ return continuation_type{hist, state}; }

      worker -> set_position(hist, state);
      worker -> go(config::init_depth);
      worker -> iterative_deepening_loop_();
      if(timer.elapsed() >= config::timeout){ return std::nullopt; }
      hist.push_(state.hash());

      last_move = worker -> best_move();
      state = state.forward(worker -> best_move());
    }

    return std::nullopt;
  }

};

}
