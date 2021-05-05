/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <istream>
#include <streambuf>

namespace nnue {

template <typename T>
struct weights_streamer {
  using signature_type = std::uint32_t;

  signature_type signature_{0};
  std::fstream reader;

  weights_streamer<T>& stream(T* dst, const size_t request) {
    constexpr size_t signature_bytes = std::min(sizeof(signature_type), sizeof(T));
    std::array<char, sizeof(T)> single_element{};
    for (size_t i(0); i < request; ++i) {
      reader.read(single_element.data(), single_element.size());
      std::memcpy(dst + i, single_element.data(), single_element.size());

      signature_type x{};
      std::memcpy(&x, single_element.data(), signature_bytes);
      signature_ ^= x;
    }
    return *this;
  }

  signature_type signature() const { return signature_; }

  weights_streamer(const std::string& name) : reader(name, std::ios_base::in | std::ios_base::binary) {}
};

template <typename T>
struct embedded_weight_streamer {
  static_assert(1 == sizeof(unsigned char), "unsigned char must be one byte wide");
  using signature_type = std::uint32_t;

  signature_type signature_{0};
  const unsigned char* back_ptr;

  embedded_weight_streamer<T>& stream(T* dst, const size_t request) {
    constexpr size_t signature_bytes = std::min(sizeof(signature_type), sizeof(T));
    std::array<unsigned char, sizeof(T)> single_element{};
    for (size_t i(0); i < request; ++i) {
      std::memcpy(single_element.data(), back_ptr, single_element.size());
      back_ptr += single_element.size();
      std::memcpy(dst + i, single_element.data(), single_element.size());

      signature_type x{};
      std::memcpy(&x, single_element.data(), signature_bytes);
      signature_ ^= x;
    }
    return *this;
  }

  signature_type signature() const { return signature_; }

  embedded_weight_streamer(const unsigned char* data) : back_ptr{data} {}
};

}  // namespace nnue
