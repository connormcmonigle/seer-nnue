#include <thread>
#include <future>

#include <uci.h>


int main(){
  engine::uci u{};
  while(true){
    std::string line{}; std::getline(std::cin, line);
    u.uci_loop(line);
  }
}
