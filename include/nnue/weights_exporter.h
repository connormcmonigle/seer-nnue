/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

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

struct weights_exporter {
  std::fstream writer;

  template <typename T>
  [[maybe_unused]] weights_exporter& write(const T* src, const std::size_t& length = static_cast<std::size_t>(1)) noexcept {
    std::array<char, sizeof(T)> single_element{};

    for (std::size_t i(0); i < length; ++i) {
      std::memcpy(single_element.data(), src + i, single_element.size());
      writer.write(single_element.data(), single_element.size());
    }

    return *this;
  }

  explicit weights_exporter(const std::string& name) noexcept : writer(name, std::ios_base::out | std::ios_base::binary) {}
};

}  // namespace nnue
