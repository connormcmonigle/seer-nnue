#include <iostream>
#include <string>
#include <cstdint>

#include <uci.h>


int main(int argc, char* argv[]){
  const std::string argument = [&]{
    if(argc <= 1){ return std::string{}; }
    std::string result{};
    for(int i(1); i < argc; ++i){
      result += argv[i];
      result += " ";
    }
    result.pop_back();
    return result;
  }();

  engine::uci u{};
  if(!argument.empty()){
    u.uci_loop(argument);
    std::exit(0);
  }
  
  while(true){
    std::string line{}; std::getline(std::cin, line);
    u.uci_loop(line);
  }
}
