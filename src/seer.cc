#include <iostream>
#include <string>

#include <uci.h>


int main(int argc, char* argv[]){
  engine::uci uci{};

  const bool perform_bench = (argc == 2) && (std::string(argv[1]) == "bench");
  if(perform_bench){ uci.bench(); return 0; }
  
  while(!uci.should_quit()){
    std::string line{}; std::getline(std::cin, line);
    uci.read(line);
  }
}
