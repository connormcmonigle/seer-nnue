#include <iostream>
#include <nnue_util.h>
#include <nnue_half_kp.h>

int main(){
  const auto weights = nnue::half_kp_weights<float>{}.load("../train/model/save.bin");
  std::cout << "w: " << weights.w.abs_max() << '\n';
  std::cout << "b: "<< weights.b.abs_max() << '\n';
  std::cout << "fc0: " << weights.fc0.abs_max() << '\n';
  std::cout << "fc1: " << weights.fc1.abs_max() << '\n';
  std::cout << "fc2: " << weights.fc2.abs_max() << '\n';
  std::cout << "skip: " << weights.skip.abs_max() << '\n';

}