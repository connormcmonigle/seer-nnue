#include <iostream>
#include <string>

#include <uci.h>


int main(int argc, char* argv[]){
  const std::string argument = [&]{
    std::string result{};
    for(int i(1); i < argc; ++i){ result += argv[i]; }
    return result;
  }();

  engine::uci u{};
  if(!argument.empty()){ u.uci_loop(argument); return 0; }

  while(true){
    std::string line{}; std::getline(std::cin, line);
    u.uci_loop(line);
  }
}
