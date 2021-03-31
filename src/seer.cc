#include <iostream>
#include <chrono>
#include <string>

#include <uci.h>


int main(int argc, char* argv[]){
  using namespace std::chrono_literals;
  engine::uci uci{};

  const bool perform_bench = (argc == 2) && (std::string(argv[1]) == "bench");
  if(perform_bench){ uci.bench(); return 0; }
  
  engine::command_loop<engine::uci>(uci, 1ms);
}
