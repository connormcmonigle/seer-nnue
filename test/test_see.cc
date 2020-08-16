#include <iostream>
#include <string>

#include <board.h>

int main(){
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  const auto bd = chess::board::parse_fen(fen);
  for(const auto& mv : bd.generate_moves()){
    std::cout << mv << std::endl;
    std::cout << bd.see<int>(mv) << std::endl;
  }
}
