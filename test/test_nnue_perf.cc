#include <iostream>
#include <chrono>
#include <cstdint>
#include <string>

#include <nnue_model.h>
#include <nnue_util.h>

template<typename T>
void time_type(const std::string& type_name){
  constexpr size_t num_runs = 10000000;
  const auto weights = nnue::weights<T>{};
  nnue::eval<T> eval(&weights);
  auto start = std::chrono::high_resolution_clock::now();
  T sum{0};
  
  for(size_t i(0); i < num_runs; ++i){
    sum += eval.propagate(true);
  }
  
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration =  std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  //prevents compiler optimization
  std::cout << "sum: " << sum << '\n';
  const double evals_per_second = static_cast<double>(num_runs) * 1e6 / static_cast<double>(duration.count());
  std::cout << "evals_per_second for " << type_name << ": " << evals_per_second << '\n';
}

int main(){
  time_type<float>("float");
  time_type<int>("int");
  time_type<std::int32_t>("int32_t");
  time_type<std::int16_t>("int16_t");
  time_type<std::int8_t>("int8_t");
}
