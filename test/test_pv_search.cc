#include <memory>
#include <iostream>

#include <nnue_half_kp.h>
#include <transposition_table.h>
#include <board.h>
#include <search.h>


int main(){
  using real_t = float;
  const auto weights = nnue::half_kp_weights<real_t>{}.load("../train/model/save.bin");

  std::shared_ptr<chess::table> tt = std::make_shared<chess::table>(1000000);
  std::cout << "enter fen: ";
  std::string fen; std::getline(std::cin, fen);
  const auto bd = chess::board::parse_fen(fen);
  const auto eval = nnue::half_kp_eval<real_t>(&weights);

  auto[score, mv] = chess::pv_search(tt, eval, bd, 4);
  std::cout << score << std::endl;
  std::cout << mv << std::endl;
}