#include <iostream>

#include <position_history.h>
#include <search_stack.h>

int main(){
  search::stack<float> stack{chess::position_history{}};
  const auto view_0 = search::stack_view<float>::root(stack);
  
  view_0.set_eval(0.1).set_hash(2).set_played(chess::move{1});
  
  const auto view_1 = view_0.next();

  view_1.set_eval(-0.2).set_hash(3);
  
  const auto view_2 = view_1.next();
  
  view_2.set_eval(0.3).set_hash(2);

  std::cout << "improving :: " << std::boolalpha << view_2.improving() << std::endl;
  std::cout << "nmp_valid :: " << std::boolalpha << view_2.nmp_valid() << std::endl;
  std::cout << "is_three_fold(2) :: " << std::boolalpha << view_2.next().is_two_fold(2) << std::endl;
}
