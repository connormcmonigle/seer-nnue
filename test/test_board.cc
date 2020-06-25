#include <iostream>
#include <string>
#include <board.h>

int main(){
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  auto bd = chess::board::parse_fen(fen);
  //auto bd = chess::board::start_pos();
  std::cout << bd << std::endl;
  for(;;){
    const chess::move_list mv_ls = bd.generate_moves();
    std::cout << mv_ls << std::endl;
    size_t i; std::cin >> i;
    bd = bd.forward(mv_ls.data[i]);
    std::cout << bd << std::endl;
  }
}
