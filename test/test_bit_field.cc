#include <iostream>
#include <cstdint>

#include <bit_field.h>
#include <enum_util.h>

int main(){
  using piece_ = chess::bit_field<chess::piece_type, 0, 3>;
  const std::uint32_t data{0};
  const auto p = piece_::get(data);
  std::cout << chess::piece_name(p) << std::endl;
}
