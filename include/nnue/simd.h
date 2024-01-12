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

// #include <x86intrin.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <type_traits>
#include <utility>

namespace simd {

[[nodiscard]] inline void* aligned_alloc(std::size_t alignment, std::size_t size) {
#if defined(_WIN32)
  return _mm_malloc(size, alignment);
#else
  return std::aligned_alloc(alignment, size);
#endif
}

inline void aligned_free(void* ptr) {
#if defined(_WIN32)
  _mm_free(ptr);
#else
  std::free(ptr);
#endif
}

constexpr std::size_t default_alignment = 16;
constexpr std::size_t alignment = default_alignment;

template <typename vector_x, typename element_type>
inline constexpr std::size_t per_unit = vector_x::size / sizeof(element_type);

template <std::size_t A, std::size_t B>
static constexpr bool divides = A % B == 0;

template <typename... Ts>
struct overload_set {};

template <typename T, typename... Ts>
struct overload_set<T, Ts...> {
  template <typename... Us>
  static auto f(Us&&... us) noexcept {
    if constexpr (T::available) {
      return T::f(std::forward<Us>(us)...);
    } else {
      return overload_set<Ts...>::f(std::forward<Us>(us)...);
    }
  }
};

template <typename T>
struct overload_set<T> {
  template <typename... Us>
  static auto f(Us&&... us) noexcept {
    return T::f(std::forward<Us>(us)...);
  }
};

template <std::size_t dim, typename T>
inline void add(T* a, const T* b) noexcept {
#pragma omp simd
  for (std::size_t i = 0; i < dim; ++i) { a[i] += b[i]; }
}

template <std::size_t dim, typename T>
inline void sub(T* a, const T* b) noexcept {
#pragma omp simd
  for (std::size_t i = 0; i < dim; ++i) { a[i] -= b[i]; }
}

template <std::size_t dim, typename T>
inline void add_add_sub(const T* a_0, const T* a_1, const T* s_0, T* out) noexcept {
#pragma omp simd
  for (std::size_t i = 0; i < dim; ++i) { out[i] = a_0[i] + a_1[i] - s_0[i]; }
}

template <std::size_t dim, typename T>
inline void add_add_sub_sub(const T* a_0, const T* a_1, const T* s_0, const T* s_1, T* out) noexcept {
#pragma omp simd
  for (std::size_t i = 0; i < dim; ++i) { out[i] = a_0[i] - s_0[i] + a_1[i] - s_1[i]; }
}

template <std::size_t dim0, std::size_t dim1, typename T0, typename T1>
inline void relu_matrix_vector_product(const T0* matrix, const T0* input, T1* output) noexcept {
#pragma omp simd
  for (std::size_t i = 0; i < dim1; ++i) {
    for (std::size_t j = 0; j < dim0; ++j) { output[i] += static_cast<T1>(std::max(input[j], T0{0})) * static_cast<T1>((matrix + i * dim0)[j]); }
  }
}

}  // namespace simd
