#include <iostream>
#include <string>

#include <board.h>
#include <position_history.h>

int main(){
  const auto bd = chess::board::start_pos();
  std::string uci_string; std::getline(std::cin, uci_string);
  auto[hist, pos] = bd.after_uci_moves(uci_string);
  std::cout << hist.occurrences(pos.hash()) << std::endl;
  std::cout << std::boolalpha << hist.is_two_fold(pos.hash()) << std::endl;
}



