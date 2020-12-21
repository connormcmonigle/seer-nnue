#pragma once
#include <fstream>
#include <random>
#include <thread>
#include <memory>
#include <atomic>
#include <mutex>
#include <optional>

#include <sample.h>
#include <move.h>
#include <nnue_model.h>
#include <transposition_table.h>
#include <search_util.h>
#include <thread_worker.h>


namespace train{

using state_type = chess::board;
using score_type = search::score_type;
constexpr score_type wdl_scale = search::wdl_scale<score_type>;

namespace config{

constexpr size_t tt_mb_size = 2;
constexpr search::depth_type continuation_depth = 4;
constexpr search::depth_type continuation_max_length = 9;

}

std::tuple<bool, search::wdl_type> terminal_check(const chess::position_history& hist, const state_type& state){
  constexpr auto win = search::wdl_type(search::wdl_scale<search::score_type>, 0, 0);
  constexpr auto draw = search::wdl_type(0, search::wdl_scale<search::score_type>, 0);
  constexpr auto loss = search::wdl_type(0, 0, search::wdl_scale<search::score_type>);

  if(hist.is_three_fold(state.hash())){ return std:tuple(true, draw); }
  if(state.generate_moves().size() == 0){  }
}

template<typename T>
struct train_interface{
  std::shared_ptr<search::constants> constants_ = std::make_shared<search::constants>(1);
  std::shared_ptr<chess::table> tt_ = std::make_shared<chess::table>(config::tt_mb_size);
  nnue::weights weights_{};

  void load_weights(const std::string& path){ weights_.load(path); }

  search::wdl_type get_wdl(const state_type& state) const {
    auto evaluator = nnue::eval(&weights_);
    state.show_init(evaluator);
    return evaluator.get_wdl(state.turn());
  }

  std::optional<state_type> get_continution(state_type state){
    const size_t man_0 = state.num_pieces();

    chess::thread_worker<T, false> worker([&worker](auto&&...){
      &weights_, tt_, constants_
      if(worker.depth() >= config::continuation_depth){ worker.stop(); }
    });

    const auto hist = chess::position_history{};

    worker.set_position(hist, state);
    worker.iterative_deepening_loop_();
    hist.push_(state.hash());

    if(!worker.best_move().is_quiet()){ return std::nullopt; }

    chess::move last_move = worker.best_move();
    state = state.forward(worker.best_move());

    for(search::depth_type length{0}; length < config::continuation_max_length; ++length){
      if(auto is_terminal = terminal_check(hist, state); std::get<bool>(is_terminal)){ return state; }
      if(last_move.is_quiet() && state.num_pieces() != man_0){ return state; }

      worker.set_position(hist, state);
      worker.iterative_deepening_loop_();
      hist.push_(state.hash());

      if(last_move.is_capture() && worker.best_move().is_quiet()){ return state; }

      chess::move last_move = worker.best_move();
      state = state.forward(worker.best_move());
    }

    return state;
  }

};

}