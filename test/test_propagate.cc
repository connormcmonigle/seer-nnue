#include <iostream>
#include <nnue_half_kp.h>

int main(){
  nnue::half_kp<float> model{};
  std::cout << model.num_parameters() << std::endl;
  model.load("../train/model/save.bin");
  
  const float result = model.propagate(true);
  std::cout << result << '\n';
}
