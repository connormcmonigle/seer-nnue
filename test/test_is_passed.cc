#include <board.h>
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
  chess::move_list mv_ls = bd.generate_moves();
  for(const auto& mv : mv_ls){
    std::cout << "mv: " << mv.name(bd.turn()) << " -> " << std::boolalpha << bd.is_passed_push(mv) << std::endl;
  }
}