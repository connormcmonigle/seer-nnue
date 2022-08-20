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

#include <type_traits>
#include <utility>

#if defined(__AVX__)
#include <immintrin.h>
#elif defined(__SSE__)
#include <xmmintrin.h>
#endif

namespace simd {

#if defined(__AVX__)
constexpr size_t alignment = 32;
#elif defined(__SSE__)
constexpr size_t alignment = 16;
#else
constexpr size_t alignment = 16;
#endif

template <typename T>
inline constexpr size_t per_unit = alignment / sizeof(T);

template <size_t A, size_t B>
static constexpr bool divides = A % B == 0;

template <typename... Ts>
struct overload_set {};

template <typename T, typename... Ts>
struct overload_set<T, Ts...> {
  template <typename... Us>
  static auto f(Us&&... us) {
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
  static auto f(Us&&... us) {
    return T::f(std::forward<Us>(us)...);
  }
};

template <size_t dim0, size_t dim1, typename T>
inline void matrix_vector_product(const T* matrix, const T* input, T* output) {
  for (size_t i(0); i < dim1; ++i) {
    for (size_t j(0); j < dim0; ++j) { output[i] += input[j] * (matrix + i * dim0)[j]; }
  }
}

#if defined(__AVX__)
template <size_t dim0, size_t dim1>
struct matrix_vector_product_x8_x1 {
  static constexpr bool available = divides<dim0, per_unit<float> >;

  static inline void f(const float* matrix, const float* input, float* output) {
    for (size_t i(0); i < dim1; ++i) {
      __m256 sum = _mm256_setzero_ps();

      for (size_t j(0); j < dim0; j += per_unit<float>) {
        const __m256 input_region = _mm256_load_ps(input + j);
        sum = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + i * dim0 + j), input_region), sum);
      }

      const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 0x1));
      const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
      const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));

      output[i] += _mm_cvtss_f32(reduced_1);
    }
  }
};

template <size_t dim0, size_t dim1>
struct matrix_vector_product_x8_x8 {
  static constexpr size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<float> >;

  static inline void f(const float* matrix, const float* input, float* output) {
    __m256* v_output = (__m256*)output;
    constexpr size_t output_step = num_units / per_unit<float>;
    for (size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m256 sum_0 = _mm256_setzero_ps();
      __m256 sum_1 = _mm256_setzero_ps();
      __m256 sum_2 = _mm256_setzero_ps();
      __m256 sum_3 = _mm256_setzero_ps();
      __m256 sum_4 = _mm256_setzero_ps();
      __m256 sum_5 = _mm256_setzero_ps();
      __m256 sum_6 = _mm256_setzero_ps();
      __m256 sum_7 = _mm256_setzero_ps();

      for (size_t j(0); j < dim0; j += per_unit<float>) {
        const __m256 input_region = _mm256_load_ps(input + j);
        sum_0 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 0) * dim0 + j), input_region), sum_0);
        sum_1 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 1) * dim0 + j), input_region), sum_1);
        sum_2 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 2) * dim0 + j), input_region), sum_2);
        sum_3 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 3) * dim0 + j), input_region), sum_3);
        sum_4 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 4) * dim0 + j), input_region), sum_4);
        sum_5 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 5) * dim0 + j), input_region), sum_5);
        sum_6 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 6) * dim0 + j), input_region), sum_6);
        sum_7 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + (i + 7) * dim0 + j), input_region), sum_7);
      }

      const __m256 sum_01 = _mm256_hadd_ps(sum_0, sum_1);
      const __m256 sum_23 = _mm256_hadd_ps(sum_2, sum_3);
      const __m256 sum_45 = _mm256_hadd_ps(sum_4, sum_5);
      const __m256 sum_67 = _mm256_hadd_ps(sum_6, sum_7);

      const __m256 sum_0123 = _mm256_hadd_ps(sum_01, sum_23);
      const __m256 sum_4567 = _mm256_hadd_ps(sum_45, sum_67);

      const __m256 sum_01234567 = _mm256_add_ps(_mm256_permute2f128_ps(sum_0123, sum_4567, 0x20), _mm256_permute2f128_ps(sum_0123, sum_4567, 0x31));

      *v_output = _mm256_add_ps(*v_output, sum_01234567);
    }
  }
};

template <size_t dim0, size_t dim1>
inline void matrix_vector_product(const float* matrix, const float* input, float* output) {
  return overload_set<matrix_vector_product_x8_x8<dim0, dim1>, matrix_vector_product_x8_x1<dim0, dim1> >::f(matrix, input, output);
}

#elif defined(__SSE__)

template <size_t dim0, size_t dim1>
struct matrix_vector_product_x8_x1 {
  static constexpr size_t num_units = 2;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<dim0, per_iteration>;

  static inline void f(const float* matrix, const float* input, float* output) {
    for (size_t i(0); i < dim1; ++i) {
      __m128 sum_0 = _mm_setzero_ps();
      __m128 sum_1 = _mm_setzero_ps();

      for (size_t j(0); j < dim0; j += per_iteration) {
        const __m128 input_region_0 = _mm_load_ps(input + j + 0 * per_unit<float>);
        const __m128 input_region_1 = _mm_load_ps(input + j + 1 * per_unit<float>);

        sum_0 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + i * dim0 + j + 0 * per_unit<float>), input_region_0), sum_0);
        sum_1 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + i * dim0 + j + 1 * per_unit<float>), input_region_1), sum_1);
      }

      const __m128 reduced_4 = _mm_add_ps(sum_0, sum_1);
      const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
      const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
      output[i] += _mm_cvtss_f32(reduced_1);
    }
  }
};

template <size_t dim0, size_t dim1>
struct matrix_vector_product_x4_x8 {
  static constexpr size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<float> >;

  static inline void f(const float* matrix, const float* input, float* output) {
    __m128* v_output = (__m128*)output;
    constexpr size_t output_step = num_units / per_unit<float>;
    for (size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m128 sum_0 = _mm_setzero_ps();
      __m128 sum_1 = _mm_setzero_ps();
      __m128 sum_2 = _mm_setzero_ps();
      __m128 sum_3 = _mm_setzero_ps();
      __m128 sum_4 = _mm_setzero_ps();
      __m128 sum_5 = _mm_setzero_ps();
      __m128 sum_6 = _mm_setzero_ps();
      __m128 sum_7 = _mm_setzero_ps();

      for (size_t j(0); j < dim0; j += per_unit<float>) {
        const __m128 input_region = _mm_load_ps(input + j);
        sum_0 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 0) * dim0 + j), input_region), sum_0);
        sum_1 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 1) * dim0 + j), input_region), sum_1);
        sum_2 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 2) * dim0 + j), input_region), sum_2);
        sum_3 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 3) * dim0 + j), input_region), sum_3);
        sum_4 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 4) * dim0 + j), input_region), sum_4);
        sum_5 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 5) * dim0 + j), input_region), sum_5);
        sum_6 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 6) * dim0 + j), input_region), sum_6);
        sum_7 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + (i + 7) * dim0 + j), input_region), sum_7);
      }

      const __m128 sum_01 = _mm_hadd_ps(sum_0, sum_1);
      const __m128 sum_23 = _mm_hadd_ps(sum_2, sum_3);
      const __m128 sum_45 = _mm_hadd_ps(sum_4, sum_5);
      const __m128 sum_67 = _mm_hadd_ps(sum_6, sum_7);

      const __m128 sum_0123 = _mm_hadd_ps(sum_01, sum_23);
      const __m128 sum_4567 = _mm_hadd_ps(sum_45, sum_67);

      *(v_output + 0) = _mm_add_ps(*(v_output + 0), sum_0123);
      *(v_output + 1) = _mm_add_ps(*(v_output + 1), sum_4567);
    }
  }
};

template <size_t dim0, size_t dim1>
inline void matrix_vector_product(const float* matrix, const float* input, float* output) {
  return overload_set<matrix_vector_product_x4_x8<dim0, dim1>, matrix_vector_product_x8_x1<dim0, dim1>>::f(matrix, input, output);
}

#endif

}  // namespace simd
