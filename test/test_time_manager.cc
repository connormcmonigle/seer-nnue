#include <iostream>

#include <time_manager.h>

int main(){
  engine::time_manager tm{};
  tm.init(true, "go wtime 1000 btime 1000 winc 100 binc 100");
  std::cout <<
    tm.get<engine::go::wtime>().value() << '\n' <<
    tm.get<engine::go::btime>().value() << '\n' <<
    tm.get<engine::go::winc>().value() << '\n' <<
    tm.get<engine::go::binc>().value() << '\n';
    
    std::cout << "\n\n\n\n";
    
    std::cout << 
      tm.min_budget.count() << '\n' <<
      tm.max_budget.count() << '\n';
}
