#include <iostream>
#include <nnue_half_kp.h>

int main(){
  const auto weights = nnue::half_kp_weights<float>{}.load("../train/model/save.bin");
  std::cout << weights.num_parameters() << std::endl;
  nnue::half_kp_eval<float> eval(&weights);
  
  const float result = eval.propagate(true);
  std::cout << result << '\n';
}
