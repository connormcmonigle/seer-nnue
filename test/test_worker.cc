#include <nnue_half_kp.h>
#include <thread_worker.h>

int main(){
  using real_t = float;
  const auto weights = nnue::half_kp_weights<real_t>{}.load("../train/model/save.bin");
  chess::worker_pool<real_t> pool(&weights, 65536, 4);
  pool.set_position(chess::board::start_pos());
  pool.go();
}

