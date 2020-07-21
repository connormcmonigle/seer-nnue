#include <iostream>
#include <thread>
#include <random>
#include <chrono>

#include <zobrist_util.h>
#include <history_heuristic.h>

int main(){
  chess::history_heuristic history{};

  auto gen = std::mt19937(std::random_device()());

  std::thread([&history, &gen](){
    auto piece = std::uniform_int_distribution<std::uint32_t>(0, 5);
    auto sq = std::uniform_int_distribution<std::uint8_t>(0, 63);

    for(;;){
      history.add(1, chess::move{chess::square::from_index(sq(gen)), chess::square::from_index(sq(gen)), static_cast<chess::piece_type>(piece(gen))});
    }
  }).detach();

  for(;;){
    std::this_thread::sleep_for(std::chrono::seconds(10));
    std::cout << history << '\n';
  }
}