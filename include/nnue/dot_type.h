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

#include <cstdint>

namespace nnue {

template <typename T>
struct dot_type_impl {};

template <>
struct dot_type_impl<float> {
  using type = float;
};

template <>
struct dot_type_impl<double> {
  using type = double;
};

template <>
struct dot_type_impl<std::int8_t> {
  using type = std::int16_t;
};

template <>
struct dot_type_impl<std::int16_t> {
  using type = std::int32_t;
};

template <>
struct dot_type_impl<std::int32_t> {
  using type = std::int64_t;
};

template <typename T>
using dot_type = typename dot_type_impl<T>::type;

}  // namespace nnue
