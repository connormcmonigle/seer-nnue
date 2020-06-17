#include <iostream>
#include <enum_util.h>

struct blah : chess::sided<blah, std::string_view> {
  std::string_view white{"white"};
  std::string_view black{"black"};
  blah(){}
};

int main(){
  const blah b{};
  std::cout << b.us<chess::color::black>() << std::endl;
  std::cout << b.them(false) << std::endl;
}
