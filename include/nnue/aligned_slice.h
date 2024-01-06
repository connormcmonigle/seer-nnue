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

#include <cstddef>
#include <cstring>
#include <iostream>

namespace nnue {

template <typename T, std::size_t dim>
struct aligned_slice {
  T* data;

  [[nodiscard]] const T* ptr() const noexcept { return data; }

  template <std::size_t out_dim, std::size_t offset = 0>
  [[nodiscard]] aligned_slice<T, out_dim> slice() noexcept {
    static_assert(offset + out_dim <= dim);
    return aligned_slice<T, out_dim>{data + offset};
  }

  [[maybe_unused]] aligned_slice<T, dim>& copy_from(const T* other) noexcept {
    std::memcpy(data, other, sizeof(T) * dim);
    return *this;
  }

  [[maybe_unused]] aligned_slice<T, dim>& copy_from(const aligned_slice<T, dim>& other) noexcept {
    std::memcpy(data, other.data, sizeof(T) * dim);
    return *this;
  }

  aligned_slice(T* data) noexcept : data{data} {}
};

template <typename T, std::size_t dim>
inline std::ostream& operator<<(std::ostream& ostr, const aligned_slice<T, dim>& vec) noexcept {
  static_assert(dim != 0, "can't stream empty slice.");
  ostr << "aligned_slice<T, " << dim << ">([";
  for (std::size_t i = 0; i < (dim - 1); ++i) { ostr << vec.data[i] << ", "; }
  ostr << vec.data[dim - 1] << "])";
  return ostr;
}

}  // namespace nnue
