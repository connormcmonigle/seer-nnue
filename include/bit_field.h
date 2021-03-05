#pragma once

#include <utility>
#include <cstdint>

namespace chess{

template<typename T, size_t B0, size_t B1>
struct bit_field{
  static_assert(B0 < B1, "wrong bit order");
  using field_type = T;
  static constexpr size_t first = B0;
  static constexpr size_t last = B1;
  
  template<typename I>
  static constexpr T get(const I& i){
    constexpr int num_bits = 8 * sizeof(I);
    static_assert(B1 < num_bits, "integral type accessed by bit_field::get has insufficient bits");
    constexpr I one = static_cast<I>(1);
    constexpr I b0 = static_cast<I>(first);
    constexpr I b1 = static_cast<I>(last);
    constexpr I mask = (one << (b1 - b0)) - one;
    return static_cast<T>((i >> b0) & mask);
  }
  
  template<typename I>
  static constexpr void set(I& i, const T& info){
    constexpr int num_bits = 8 * sizeof(I);
    static_assert(B1 < num_bits, "integral type accessed by bit_field::set has insufficient bits");
    constexpr I one = static_cast<I>(1);
    constexpr I b0 = static_cast<I>(first);
    constexpr I b1 = static_cast<I>(last);
    const I info_ = static_cast<I>(info);
    constexpr I mask = ((one << (b1 - b0)) - one) << b0;
    i &= ~mask;
    i |= (info_ << b0) & mask;
  }
};

}
