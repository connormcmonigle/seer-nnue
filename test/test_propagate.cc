#include <iostream>

#include <nnue_model.h>
#include <board.h>

int main(){
  const auto weights = nnue::weights<float>{}.load("../train/model/save.bin");
  std::cout << "numel: " << weights.num_parameters() << std::endl;
  std::cout << "signature: 0x" << std::hex << weights.signature() << '\n';

  nnue::eval<float> eval(&weights);
  
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
    eval = bd.apply_update(mv_ls.data[i], eval);
    bd = bd.forward(mv_ls.data[i]);
  }
}
