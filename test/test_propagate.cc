#include <iostream>

#include <nnue_half_kp.h>
#include <board.h>

int main(){
  const auto weights = nnue::half_kp_weights<float>{}.load("../train/model/save.bin");
  std::cout << weights.num_parameters() << std::endl;
  nnue::half_kp_eval<float> eval(&weights);
  
  std::cout << "fen: ";
  std::string fen; std::getline(std::cin, fen);
  auto bd = chess::board::parse_fen(fen);
  bd.show_init(eval);
  
  for(;;){
    const float result = eval.propagate(bd.turn());
    std::cout << bd.fen();
    std::cout << " -> real: " << result << '\n';
    
    const chess::move_list mv_ls = bd.generate_moves();
    std::cout << mv_ls << std::endl;
    size_t i; std::cin >> i;
    eval = bd.half_kp_updated(mv_ls.data[i], eval);
    bd = bd.forward(mv_ls.data[i]);
  }
}
