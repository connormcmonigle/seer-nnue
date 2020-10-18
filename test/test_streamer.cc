#include <iostream>
#include <vector>

#include <nnue_half_kp.h>

int main(){
  const size_t numel = nnue::half_kp_weights<float>{}.num_parameters();
  std::vector<float> weights(numel);
  nnue::weights_streamer<float>("../train/model/save.bin").stream(weights.data(), numel);
  for(const float& elem : weights){ std::cout << elem << '\n'; }
}
