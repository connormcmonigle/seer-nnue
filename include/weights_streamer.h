#pragma once
#include <fstream>
#include <cstring>
#include <string>

namespace nnue{

template<typename T>
struct weights_streamer{
  std::fstream file;
  
  weights_streamer<T>& stream(T* dst, const size_t request){
    const size_t stream_size = sizeof(T) * request;
    char* char_ptr = new char[stream_size];
    file.read(char_ptr, stream_size);
    std::memcpy(dst, char_ptr, stream_size);
    delete[] char_ptr;
    return *this;
  }
  
  weights_streamer(const std::string& name) : file(name, std::ios_base::in | std::ios_base::binary) {}
};

}
