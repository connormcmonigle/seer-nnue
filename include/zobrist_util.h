#pragma once

#include <limits>
#include <cstdint>
#include <random>

namespace zobrist{

using hash_type = std::uint64_t;
constexpr unsigned int seed = 0;
  
inline hash_type random_bit_string(){
  static std::mt19937 gen{seed};
  constexpr hash_type a = std::numeric_limits<hash_type>::min();
  constexpr hash_type b = std::numeric_limits<hash_type>::max();
  std::uniform_int_distribution<hash_type> dist(a, b);
  return dist(gen);
}

}
