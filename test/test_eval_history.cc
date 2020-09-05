#include <iostream>
#include <position_history.h>

int main(){
  chess::eval_history<float> hist{};
  hist.push_(0.3).push_(0.5).push_(0.4).push_(-0.8);
  std::cout << std::boolalpha << hist.improving() << std::endl;
}
