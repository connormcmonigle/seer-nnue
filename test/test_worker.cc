#include <iostream>
#include <thread>
#include <chrono>

#include <position_history.h>
#include <board.h>
#include <nnue_half_kp.h>
#include <thread_worker.h>


int main(){
  using real_t = float;
  const auto weights = nnue::half_kp_weights<real_t>{}.load("../train/model/save.bin");
  chess::worker_pool<real_t> pool(&weights, 2048, 4);
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  pool.set_position(chess::position_history{}, chess::board::parse_fen(fen));
  pool.go();
  while(true){
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    std::cout << pool.pool_[0] -> score() << '\n';
    std::cout << pool.pool_[0] -> best_move() << '\n';
    std::cout << pool.hh_ -> white << '\n';
  }
}

