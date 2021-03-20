#include <iostream>
#include <string>

#include <uci.h>


int main(int argc, char* argv[]){
  engine::uci u{};

  const bool perform_bench = (argc == 2) && (std::string(argv[1]) == "bench");
  if(perform_bench){ u.bench(); return 0; }
  
  while(!u.should_quit()){
    std::string line{}; std::getline(std::cin, line);
    u.uci_loop(line);
  }
}
