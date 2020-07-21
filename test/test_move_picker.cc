#include <iostream>
#include <string>

#include <board.h>
#include <move.h>
#include <history_heuristic.h>
#include <move_picker.h>

int main(){
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  const auto bd = chess::board::parse_fen(fen);
  chess::history_heuristic hh{};
  chess::move_picker picker(bd.generate_moves(), &hh);
  while(!picker.empty()){
    std::cout << picker.pick() << '\n';
  }
}