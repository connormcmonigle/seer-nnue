#include <iostream>
#include <board.h>

int main(){
  std::string fen;
  std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  auto bd = chess::board::parse_fen(fen);
  for(;;){
    std::cout << bd.fen() << std::endl;
    std::cout << bd.hash() << " :match: ";
    std::cout << chess::board::parse_fen(bd.fen()).hash() << std::endl;
    const chess::move_list mv_ls = bd.generate_moves();
    std::cout << mv_ls << std::endl;
    size_t i; std::cin >> i;
    bd = bd.forward(mv_ls.data[i]);
  }
}

