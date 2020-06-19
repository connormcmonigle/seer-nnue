#include <iostream>
#include <string>
#include <board.h>

int main(){
  std::string fen;
  std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  auto bd = chess::board::parse_fen(fen);
  std::cout << bd << std::endl;
}
