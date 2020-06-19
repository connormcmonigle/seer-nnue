#include <iostream>
#include <utility>
#include <chrono>

#include <board.h>

size_t perft(const chess::board& bd, int depth){
  const auto ls = bd.generate_moves();
  if(depth == 0){
    return ls.size();
  }else{
    size_t sum = 0;
    for(size_t i(0); i < ls.size(); ++i){
      sum += perft(bd.forward(ls.data[i]), depth - 1);
    }
    return sum;
  }
}

template<typename ... Ts>
size_t perft_timed(Ts&& ... ts){
  auto start = std::chrono::high_resolution_clock::now(); 
  size_t result = perft(std::forward<Ts>(ts)...);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =  std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  const auto mnps = static_cast<double>(result) / static_cast<double>(duration.count());
  std::cout << mnps << "Mnps\n";
  return result;
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
        const size_t count = perft(bd.forward(ls.data[i]), depth - 1);
        std::cout << count << std::endl;
        sum += count;
      }
      std::cout << "total: " << sum << std::endl;
      int choice; std::cin >> choice;
      bd = bd.forward(ls.data[choice]);
    }
  }
}

int main(){
  int depth = 6;
  auto bd = chess::board::start_pos();
  std::cout << perft_timed(bd, depth) << std::endl;
}
