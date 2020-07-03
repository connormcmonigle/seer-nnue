#include <iostream>

#include <option_parser.h>

int main(){
  auto one = engine::option_callback(engine::string_option("one"), [](const std::string& a){
     std::cout << a << std::endl;
  });
  
  auto two = engine::option_callback(engine::spin_option("two", 0, engine::spin_range{0, 100}), [](const int a){
     std::cout << a << std::endl;
  });

  auto three = engine::option_callback(engine::string_option("three", "hello"), [](const std::string a){
     std::cout << a << std::endl;
  });

  auto four = engine::option_callback(engine::button_option("four"), [](bool){
    std::cout << "button pushed" << std::endl;
  });

  auto parser = engine::uci_options(one, two, three, four);
  std::cout << parser << std::endl;
  while(true){
    std::string line; std::getline(std::cin, line);
    parser.update(line);
  }
}