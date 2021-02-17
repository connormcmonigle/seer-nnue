#include <iostream>
#include <random>
#include <exception>

#include <board.h>

int main(){
  std::cout << "enter fen:";
  std::string fen;
  std::getline(std::cin, fen);
  std::cout << fen << std::endl;
  auto gen = std::mt19937(std::random_device()());
  std::cout << "enter number of runs: ";
  size_t num_runs; std::cin >> num_runs;
  for(size_t run(0); run < num_runs; ++run){
    auto bd = chess::board::parse_fen(fen);
    for(;;){
      std::cout << bd.fen() << std::endl;
      const auto h0 = bd.hash();
      const auto bd_from_bd = chess::board::parse_fen(bd.fen());
      const auto h1 = bd_from_bd.hash();

      std::cout << h0 << " :match: " << h1 << std::endl;
      
      if(h0 != h1){
        std::cout << bd_from_bd << std::endl;
        std::cout << "FAIL" << std::endl;
        std::terminate();
      }
      
      if(!bd.is_check()){
        const auto h2 = bd.forward(chess::move::null()).hash();
        if(h0 == h2){
          std::cout << bd_from_bd << std::endl;
          std::cout << "FAIL - null move" << std::endl;
          std::terminate();
        }
      }
      
      if(bd.mirrored().mirrored().hash() != h0){
        std::cout << "FAIL - mirror" << std::endl;
        std::terminate();
      } 

      const chess::move_list mv_ls = bd.generate_moves();
      if(mv_ls.size() == 0 || bd.lat_.move_count > 500){
        std::cout << "SUCCESS" << std::endl;
        break;
      }
      std::uniform_int_distribution<size_t> dist(0, mv_ls.size() - 1);
      const size_t i = dist(gen);
      std::cout << "\n\n" << i << "\n\n";
      bd = bd.forward(mv_ls.data[i]);
    }
  }
}

