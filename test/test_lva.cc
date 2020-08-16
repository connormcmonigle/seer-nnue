#include <iostream>

#include <board.h>

int main(){
  const auto tgt = chess::tbl_square{3, 3}.to_square();
  std::cout << tgt.name() << '\n';
  std::string fen; std::getline(std::cin, fen);
  const auto bd = chess::board::parse_fen(fen);
  const auto[p, sq] = bd.least_valuable_attacker<chess::color::white>(tgt, chess::square_set{});
  std::cout << chess::piece_name(p) << '\n' << sq << '\n';
}
