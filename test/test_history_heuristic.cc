#include <iostream>
#include <random>
#include <chrono>

#include <zobrist_util.h>
#include <history_heuristic.h>

int main(){
  auto gen = std::mt19937(std::random_device()());
  
  auto rnd_move = [&gen](){
    auto piece = std::uniform_int_distribution<std::uint32_t>(0, 5);
    auto sq = std::uniform_int_distribution<std::uint8_t>(0, 63);
    return chess::move{chess::square::from_index(sq(gen)), chess::square::from_index(sq(gen)), static_cast<chess::piece_type>(piece(gen))};
  };

  chess::history_heuristic history{};
  for(size_t i(0); i < 600000; ++i){
    history.update(rnd_move(), rnd_move(), rnd_move(), chess::move_list{}, 1);
  }
  std::cout << history << std::endl;
}
