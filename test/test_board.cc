#include <iostream>
#include <string>
#include <board.h>

int main(){
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  auto bd = chess::board::parse_fen(fen);
  std::cout << bd << std::endl;
  std::cout << bd.fen() << std::endl;
  std::cout << bd.mirrored().fen() << std::endl;
  
  for(;;){
    const chess::move_list mv_ls = bd.generate_moves();
    std::cout << "noisy:\n" << bd.generate_loud_moves() << std::endl;
    std::cout << "all:\n" << bd.generate_moves() << std::endl;
    size_t i; std::cin >> i;
    bd = bd.forward(mv_ls.data[i]);
    std::cout << bd << std::endl;
    std::cout << bd.fen() << std::endl;
    std::cout << bd.mirrored().fen() << std::endl;
  }
}
