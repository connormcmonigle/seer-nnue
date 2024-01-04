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

#include <nnue/simd.h>

#include <cmath>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <utility>

namespace nnue {

template <typename T, std::size_t dim>
struct aligned_vector {
  static constexpr std::size_t dimension = dim;
  alignas(simd::alignment) T data[dim];

  template <typename F>
  [[nodiscard]] constexpr aligned_vector<T, dim> apply(F&& f) const noexcept {
    return aligned_vector<T, dim>{*this}.apply_(std::forward<F>(f));
  }

  template <typename F>
  [[maybe_unused]] inline aligned_vector<T, dim>& apply_(F&& f) noexcept {
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { data[i] = f(data[i]); }
    return *this;
  }

  [[maybe_unused]] inline aligned_vector<T, dim>& softmax_() noexcept {
    static_assert(dim != 0, "can't softmax empty vector.");
    T maximum_value = data[0];
    for (std::size_t i = 0; i < dim; ++i) {
      if (data[i] > maximum_value) { maximum_value = data[i]; }
    }
    apply_([maximum_value](const T& x) { return std::exp(x - maximum_value); });
    const T z = sum();
    apply_([z](const T& x) { return x / z; });
    return *this;
  }

  [[maybe_unused]] inline aligned_vector<T, dim>& add_(const T* other) noexcept {
    simd::add<dim>(data, other);
    return *this;
  }

  [[maybe_unused]] inline aligned_vector<T, dim>& sub_(const T* other) noexcept {
    simd::sub<dim>(data, other);
    return *this;
  }

  [[maybe_unused]] inline aligned_vector<T, dim>& set_(const T* other) noexcept {
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { data[i] = other[i]; }
    return *this;
  }

  [[nodiscard]] inline aligned_slice<T, dim> as_slice() noexcept { return aligned_slice<T, dim>(data); }

  [[nodiscard]] inline T item() const noexcept {
    static_assert(dim == 1, "called item() on vector with dim != 1");
    return data[0];
  }

  [[nodiscard]] inline T sum() const noexcept {
    T result{};
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { result += data[i]; }
    return result;
  }

  template <typename U>
  [[nodiscard]] inline aligned_vector<U, dim> dequantized(const U& scale) const noexcept {
    static_assert(std::is_integral_v<T> && std::is_floating_point_v<U>);
    aligned_vector<U, dim> result;
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { result.data[i] = scale * static_cast<U>(data[i]); }
    return result;
  }

  [[nodiscard]] static inline aligned_vector<T, dim> zeros() noexcept {
    aligned_vector<T, dim> result{};
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { result.data[i] = T(0); }
    return result;
  }

  [[nodiscard]] static inline aligned_vector<T, dim> ones() noexcept {
    aligned_vector<T, dim> result{};
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { result.data[i] = T(1); }
    return result;
  }

  [[nodiscard]] static inline aligned_vector<T, dim> from(const T* data) noexcept {
    aligned_vector<T, dim> result{};
#pragma omp simd
    for (std::size_t i = 0; i < dim; ++i) { result.data[i] = data[i]; }
    return result;
  }
};

template <typename T, std::size_t dim0, std::size_t dim1>
[[nodiscard]] inline aligned_vector<T, dim0 + dim1> concat(const aligned_vector<T, dim0>& a, const aligned_vector<T, dim1>& b) noexcept {
  auto c = aligned_vector<T, dim0 + dim1>::zeros();
#pragma omp simd
  for (std::size_t i = 0; i < dim0; ++i) { c.data[i] = a.data[i]; }
  for (std::size_t i = 0; i < dim1; ++i) { c.data[dim0 + i] = b.data[i]; }
  return c;
}

template <typename T, std::size_t dim>
inline std::ostream& operator<<(std::ostream& ostr, const aligned_vector<T, dim>& vec) noexcept {
  static_assert(dim != 0, "can't stream empty vector.");
  ostr << "aligned_vector<T, " << dim << ">([";
  for (std::size_t i = 0; i < (dim - 1); ++i) { ostr << vec.data[i] << ", "; }
  ostr << vec.data[dim - 1] << "])";
  return ostr;
}

}  // namespace nnue