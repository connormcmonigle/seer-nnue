#include <future>
#include <thread>
#include <fstream>
#include <uci.h>

std::string read_line(){
  std::fstream log("/home/connor/seer-nnue/build/log.txt", std::ios::app);
  std::string line{};
  std::getline(std::cin, line);
  log << line << '\n';
  return line;
}

int main(){
  const auto weights = nnue::half_kp_weights<engine::uci::real_t>{}.load("/home/connor/seer-nnue/train/model/save.bin");
  engine::uci u(/*weights=*/&weights, /*hash_size=*/4000, /*thread_count=*/4);
  std::future<std::string> future = std::async(read_line);
  while(true){
    if(future.wait_for(std::chrono::seconds(0)) == std::future_status::ready){
      u.uci_loop(future.get());
      future = std::async(read_line);
    }else{
      u.uci_loop("");
    }
  }
}
