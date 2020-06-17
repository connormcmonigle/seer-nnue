#include <board.h>

size_t perft(const chess::board& bd, int depth){
  const auto ls = bd.generate_moves();
  if(depth == 0){
    return ls.size();
  }else{
    size_t sum = 0;
    for(size_t i(0); i < ls.size(); ++i){
      chess::board bd_copy = bd;
      bd_copy.forward(ls.data[i]);
      sum += perft(bd_copy, depth - 1);
    }
    return sum;
  }
}

void perft_divide(chess::board bd, int depth){
  for(; depth >= 0; --depth){
    const auto ls = bd.generate_moves();
    if(depth == 0){
      std::cout << ls << std::endl;
      std::cout << bd << std::endl;
    }else{
      size_t sum{0};
      for(size_t i(0); i < ls.size(); ++i){
        std::cout << i << ". " << ls.data[i] << " -> ";
        chess::board bd_copy = bd;
        bd_copy.forward(ls.data[i]);
        const size_t count = perft(bd_copy, depth - 1);
        std::cout << count << std::endl;
        sum += count;
      }
      std::cout << "total: " << sum << std::endl;
      int choice; std::cin >> choice;
      bd.forward(ls.data[choice]);
    }
  }
}

int main(){
  int depth; std::cin >> depth;
  auto bd = chess::board::parse_fen("r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10 ");
  perft_divide(bd, depth);
}
