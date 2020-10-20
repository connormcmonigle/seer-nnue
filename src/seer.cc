#include <thread>
#include <future>

#include <uci.h>

std::string read_line(){
  std::string line{};
  std::getline(std::cin, line);
  return line;
}

int main(){
  engine::uci u{};
  std::future<std::string> future = std::async(read_line);
  while(true){
    if(!u.searching()){ future.wait(); }
    if(future.wait_for(std::chrono::milliseconds(1)) == std::future_status::ready){
      u.uci_loop(future.get());
      future = std::async(read_line);
    }else{
      u.uci_loop("");
    }
  }
}
