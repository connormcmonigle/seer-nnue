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

#include <nnue/aligned_slice.h>
#include <nnue/simd.h>

#include <cstddef>
#include <cstring>

namespace nnue {

template <typename T, std::size_t scratchpad_size>
struct aligned_scratchpad {
  alignas(simd::alignment) T data[scratchpad_size];

  template <std::size_t dim>
  [[nodiscard]] aligned_slice<T, dim> get_nth_slice(const std::size_t& n) noexcept {
    static_assert(scratchpad_size % dim == 0);
    return aligned_slice<T, dim>(data + n * dim);
  }
};

}  // namespace nnue
