#pragma once
#include <fstream>
#include <cstring>
#include <string>

#include <array>

namespace nnue{

template<typename T>
struct weights_streamer{
  std::fstream file;
  
  weights_streamer<T>& stream(T* dst, const size_t request){
    std::array<char, sizeof(T)> single_element{};
    for(size_t i(0); i < request; ++i){
      file.read(single_element.data(), single_element.size());
      std::memcpy(dst + i, single_element.data(), single_element.size());
    }
    return *this;
  }
  
  weights_streamer(const std::string& name) : file(name, std::ios_base::in | std::ios_base::binary) {}
};

}
