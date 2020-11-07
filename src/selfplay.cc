#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>

#include <sample.h>
#include <search_util.h>
#include <nnue_model.h>
#include <data_generator.h>

int main(){
  using real_t = float;
  std::cout << "weights path: ";
  std::string weights_path{}; std::cin >> weights_path;
  std::cout << "num_threads: ";
  size_t num_threads{}; std::cin >> num_threads;
  std::cout << "depth: ";
  search::depth_type depth{}; std::cin >> depth;
  std::cout << "num_positions: ";
  size_t num_positions{}; std::cin >> num_positions;
  std::cout << "destination_path: ";
  std::string destination_path{}; std::cin >> destination_path;
  
  auto destination = std::fstream(destination_path);
  const auto weights = nnue::weights<real_t>{}.load(weights_path);
  
  selfplay::generator_config<real_t, selfplay::default_random_inserter> cfg(&weights, depth);
  selfplay::generator<real_t, selfplay::default_random_inserter> generator(cfg, num_threads, num_positions);
  const auto start = std::chrono::high_resolution_clock::now();
  for(;;){
    std::cout << '\r';
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
    generator.poll(destination);
    
    const auto present =std::chrono::high_resolution_clock::now();
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(present - start).count();
    
    std::cout << "\r" << "total written: " << generator.total_written  << std::flush << ", positions per second: " << (generator.total_written * 1000 / elapsed);
  }
  
}
