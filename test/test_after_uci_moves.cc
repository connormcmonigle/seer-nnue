#include <iostream>
#include <string>
#include <board.h>

int main(){
  const auto bd = chess::board::start_pos();
  std::string uci_string; std::getline(std::cin, uci_string);
  std::cout << bd.after_uci_moves(uci_string) << '\n';
}