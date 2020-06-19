#include <iostream>
#include <random>
#include <board.h>

int main(){
  std::string fen;
  std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  auto gen = std::mt19937(std::random_device()());
  auto bd = chess::board::parse_fen(fen);
  for(;;){
    std::cout << bd.fen() << std::endl;
    const auto h0 = bd.hash();
    const auto bd_from_bd = chess::board::parse_fen(bd.fen());
    const auto h1 = bd_from_bd.hash();
    std::cout << h0 << " :match: " << h1 << std::endl;
    if(h0 != h1){
      std::cout << bd_from_bd << std::endl;
      std::cout << "FAIL" << std::endl;
      break;
    }
    const chess::move_list mv_ls = bd.generate_moves();
    if(mv_ls.size() == 0 || bd.lat_.move_count > 500){
      std::cout << "SUCCESS" << std::endl;
      break;
    }
    std::uniform_int_distribution<size_t> dist(0, mv_ls.size() - 1);
    const size_t i = dist(gen);
    std::cout << "\n\n" << i << "\n\n";
    bd = bd.forward(mv_ls.data[i]);
  }
}

