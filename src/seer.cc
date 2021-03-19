#include <iostream>
#include <string>
#include <cstdint>

#include <uci.h>


int main(int argc, char* argv[]){
  engine::uci u{};

  const bool bench = (argc == 2) && (std::string(argv[1]) == "bench");
  if(bench){ u.bench(); std::exit(0); }
  
  while(true){
    std::string line{}; std::getline(std::cin, line);
    u.uci_loop(line);
  }
}
