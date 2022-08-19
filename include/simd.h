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

#if defined(__AVX512DQ__)
constexpr size_t alignment = 64;
#elif defined(__AVX__)
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
  template <typename ... Us>
  static auto f(Us&& ... us) {
    if constexpr (T::available) {
      return T::f(std::forward<Us>(us)...);
    } else {
      return overload_set<Ts...>::f(std::forward<Us>(us)...);
    }
  }
};

template <typename T>
struct overload_set<T> {
  template <typename ... Us>
  static auto f(Us&& ... us) {
    return T::f(std::forward<Us>(us)...);
  }
};

template <size_t N, typename T>
inline T dot_product(const T* a, const T* b) {
  T sum{};
#pragma omp simd
  for (size_t i = 0; i < N; ++i) { sum += a[i] * b[i]; }
  return sum;
}

template <size_t dim0, size_t dim1, typename T>
inline void matrix_vector_product(const T* matrix, const T* input, T* output) {
  for (size_t i(0); i < dim1; ++i) { output[i] += dot_product<dim0, T>(input, matrix + i * dim0); }
}

#if defined(__AVX512DQ__)

template <size_t N>
struct dot_product_8_type {
  static constexpr size_t num_units = 1;
  static constexpr size_t per_iteration = 8;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      const __m256 a_0 = _mm256_load_ps(a);
      const __m256 b_0 = _mm256_load_ps(b);
      sum = _mm256_fmadd_ps(a_0, b_0, sum);
    }

    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const float res = _mm_cvtss_f32(reduced_1);
    return res;
  }
};

template <size_t N>
struct dot_product_16_type {
  static constexpr size_t num_units = 1;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m512 sum = _mm512_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      const __m512 a_0 = _mm512_load_ps(a);
      const __m512 b_0 = _mm512_load_ps(b);
      sum = _mm512_fmadd_ps(a_0, b_0, sum);
    }

    const __m256 reduced_8 = _mm256_add_ps(_mm512_castps512_ps256(sum), _mm512_extractf32x8_ps(sum, 1));
    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 1));
    const float res = _mm_cvtss_f32(reduced_1);
    return res;
  }
};

template <size_t N>
struct dot_product_32_type {
  static constexpr size_t num_units = 2;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m512 sum_0 = _mm512_setzero_ps();
    __m512 sum_1 = _mm512_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      {
        const __m512 a_0 = _mm512_load_ps(a + 0 * per_unit<float>);
        const __m512 b_0 = _mm512_load_ps(b + 0 * per_unit<float>);
        sum_0 = _mm512_fmadd_ps(a_0, b_0, sum_0);
      }

      {
        const __m512 a_1 = _mm512_load_ps(a + 1 * per_unit<float>);
        const __m512 b_1 = _mm512_load_ps(b + 1 * per_unit<float>);
        sum_1 = _mm512_fmadd_ps(a_1, b_1, sum_1);
      }
    }

    const __m512 reduced_16 = _mm512_add_ps(sum_0, sum_1);
    const __m256 reduced_8 = _mm256_add_ps(_mm512_castps512_ps256(reduced_16), _mm512_extractf32x8_ps(reduced_16, 1));
    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 1));
    const float sum = _mm_cvtss_f32(reduced_1);
    return sum;
  }
};

template <size_t N>
struct dot_product_64_type {
  static constexpr size_t num_units = 4;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m512 sum_0 = _mm512_setzero_ps();
    __m512 sum_1 = _mm512_setzero_ps();
    __m512 sum_2 = _mm512_setzero_ps();
    __m512 sum_3 = _mm512_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      {
        const __m512 a_0 = _mm512_load_ps(a + 0 * per_unit<float>);
        const __m512 b_0 = _mm512_load_ps(b + 0 * per_unit<float>);
        sum_0 = _mm512_fmadd_ps(a_0, b_0, sum_0);
      }

      {
        const __m512 a_1 = _mm512_load_ps(a + 1 * per_unit<float>);
        const __m512 b_1 = _mm512_load_ps(b + 1 * per_unit<float>);
        sum_1 = _mm512_fmadd_ps(a_1, b_1, sum_1);
      }

      {
        const __m512 a_2 = _mm512_load_ps(a + 2 * per_unit<float>);
        const __m512 b_2 = _mm512_load_ps(b + 2 * per_unit<float>);
        sum_2 = _mm512_fmadd_ps(a_2, b_2, sum_2);
      }

      {
        const __m512 a_3 = _mm512_load_ps(a + 3 * per_unit<float>);
        const __m512 b_3 = _mm512_load_ps(b + 3 * per_unit<float>);
        sum_3 = _mm512_fmadd_ps(a_3, b_3, sum_3);
      }
    }

    const __m512 reduced_16 = _mm512_add_ps(_mm512_add_ps(sum_0, sum_1), _mm512_add_ps(sum_2, sum_3));
    const __m256 reduced_8 = _mm256_add_ps(_mm512_castps512_ps256(reduced_16), _mm512_extractf32x8_ps(reduced_16, 1));
    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 1));
    const float sum = _mm_cvtss_f32(reduced_1);
    return sum;
  }
};

template <size_t N>
inline float dot_product(const float* a, const float* b) {
  return overload_set<dot_product_64_type<N>, dot_product_32_type<N>, dot_product_16_type<N>, dot_product_8_type<N> >::f(a, b);
}

template <size_t dim0, size_t dim1>
inline void matrix_vector_product(const float* matrix, const float* input, float* output) {
  using dot_product_type =
      overload_set<dot_product_64_type<dim0>, dot_product_32_type<dim0>, dot_product_16_type<dim0>, dot_product_8_type<dim0> >;
  for (size_t i(0); i < dim1; ++i) { output[i] += dot_product_type::f(input, matrix + i * dim0); }
}

#elif defined(__AVX__)

template <size_t N>
struct dot_product_8_type {
  static constexpr size_t num_units = 1;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m256 sum = _mm256_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      const __m256 a_0 = _mm256_load_ps(a);
      const __m256 b_0 = _mm256_load_ps(b);
      sum = _mm256_add_ps(_mm256_mul_ps(a_0, b_0), sum);
    }

    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const float res = _mm_cvtss_f32(reduced_1);
    return res;
  }
};

template <size_t N>
struct dot_product_16_type {
  static constexpr size_t num_units = 2;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m256 sum_0 = _mm256_setzero_ps();
    __m256 sum_1 = _mm256_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      {
        const __m256 a_0 = _mm256_load_ps(a + 0 * per_unit<float>);
        const __m256 b_0 = _mm256_load_ps(b + 0 * per_unit<float>);
        sum_0 = _mm256_add_ps(_mm256_mul_ps(a_0, b_0), sum_0);
      }

      {
        const __m256 a_1 = _mm256_load_ps(a + 1 * per_unit<float>);
        const __m256 b_1 = _mm256_load_ps(b + 1 * per_unit<float>);
        sum_1 = _mm256_add_ps(_mm256_mul_ps(a_1, b_1), sum_1);
      }
    }

    const __m256 reduced_8 = _mm256_add_ps(sum_0, sum_1);
    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const float sum = _mm_cvtss_f32(reduced_1);
    return sum;
  }
};

template <size_t N>
struct dot_product_32_type {
  static constexpr size_t num_units = 4;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m256 sum_0 = _mm256_setzero_ps();
    __m256 sum_1 = _mm256_setzero_ps();
    __m256 sum_2 = _mm256_setzero_ps();
    __m256 sum_3 = _mm256_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      {
        const __m256 a_0 = _mm256_load_ps(a + 0 * per_unit<float>);
        const __m256 b_0 = _mm256_load_ps(b + 0 * per_unit<float>);
        sum_0 = _mm256_add_ps(_mm256_mul_ps(a_0, b_0), sum_0);
      }

      {
        const __m256 a_1 = _mm256_load_ps(a + 1 * per_unit<float>);
        const __m256 b_1 = _mm256_load_ps(b + 1 * per_unit<float>);
        sum_1 = _mm256_add_ps(_mm256_mul_ps(a_1, b_1), sum_1);
      }

      {
        const __m256 a_2 = _mm256_load_ps(a + 2 * per_unit<float>);
        const __m256 b_2 = _mm256_load_ps(b + 2 * per_unit<float>);
        sum_2 = _mm256_add_ps(_mm256_mul_ps(a_2, b_2), sum_2);
      }

      {
        const __m256 a_3 = _mm256_load_ps(a + 3 * per_unit<float>);
        const __m256 b_3 = _mm256_load_ps(b + 3 * per_unit<float>);
        sum_3 = _mm256_add_ps(_mm256_mul_ps(a_3, b_3), sum_3);
      }
    }

    const __m256 reduced_8 = _mm256_add_ps(_mm256_add_ps(sum_0, sum_1), _mm256_add_ps(sum_2, sum_3));
    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const float sum = _mm_cvtss_f32(reduced_1);
    return sum;
  }
};

template <size_t dim0, size_t dim1>
struct matrix_vector_product_fallback_type {
  static constexpr bool available = divides<dim0, per_unit<float>>;

  static inline void f(const float* matrix, const float* input, float* output) {
    using dot_product_type = overload_set<dot_product_32_type<dim0>, dot_product_16_type<dim0>, dot_product_8_type<dim0> >;
    for (size_t i(0); i < dim1; ++i) { output[i] += dot_product_type::f(input, matrix + i * dim0); }    
  }
};

template <size_t dim0, size_t dim1>
struct matrix_vector_product_optimized_type {
  static constexpr bool available = dim1 == 8 && divides<dim0, per_unit<float>>;

  static inline void f(const float* matrix, const float* input, float* output) {
    __m256 sum_0 = _mm256_setzero_ps();
    __m256 sum_1 = _mm256_setzero_ps();
    __m256 sum_2 = _mm256_setzero_ps();
    __m256 sum_3 = _mm256_setzero_ps();
    __m256 sum_4 = _mm256_setzero_ps();
    __m256 sum_5 = _mm256_setzero_ps();
    __m256 sum_6 = _mm256_setzero_ps();
    __m256 sum_7 = _mm256_setzero_ps();

    for (size_t i(0); i < dim0; i += per_unit<float>) {
        const __m256 input_region = _mm256_load_ps(input + i);
        sum_0 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 0 * dim0 + i), input_region), sum_0);
        sum_1 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 1 * dim0 + i), input_region), sum_1);
        sum_2 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 2 * dim0 + i), input_region), sum_2);
        sum_3 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 3 * dim0 + i), input_region), sum_3);
        sum_4 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 4 * dim0 + i), input_region), sum_4);
        sum_5 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 5 * dim0 + i), input_region), sum_5);
        sum_6 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 6 * dim0 + i), input_region), sum_6);
        sum_7 = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + 7 * dim0 + i), input_region), sum_7);
    }


    const __m256 sum_0_1 = _mm256_hadd_ps(sum_0, sum_1);
    const __m256 sum_2_3 = _mm256_hadd_ps(sum_2, sum_3);
    const __m256 sum_4_5 = _mm256_hadd_ps(sum_4, sum_5);
    const __m256 sum_6_7 = _mm256_hadd_ps(sum_6, sum_7);

    const __m256 sum_0_1_2_3 = _mm256_hadd_ps(sum_0_1, sum_2_3);
    const __m256 sum_4_5_6_7 = _mm256_hadd_ps(sum_4_5, sum_6_7);

    const __m128 sum_abcd_0 = _mm256_extractf128_ps(sum_0_1_2_3, 0);
    const __m128 sum_abcd_1 = _mm256_extractf128_ps(sum_0_1_2_3, 1);
    const __m128 sum_efgh_0 = _mm256_extractf128_ps(sum_4_5_6_7, 0);
    const __m128 sum_efgh_1 = _mm256_extractf128_ps(sum_4_5_6_7, 1);

    const __m128 sum_abcd = _mm_add_ps(sum_abcd_0, sum_abcd_1);
    const __m128 sum_efgh = _mm_add_ps(sum_efgh_0, sum_efgh_1);

    const __m256 total = _mm256_insertf128_ps(_mm256_castps128_ps256(sum_abcd), sum_efgh, 1);

    __m256* v_output = (__m256*) output;
    *v_output = _mm256_add_ps(*v_output, total);
  }
};

template <size_t N>
inline float dot_product(const float* a, const float* b) {
  return overload_set<dot_product_32_type<N>, dot_product_16_type<N>, dot_product_8_type<N> >::f(a, b);
}

template <size_t dim0, size_t dim1>
inline void matrix_vector_product(const float* matrix, const float* input, float* output) {
  return overload_set<matrix_vector_product_optimized_type<dim0, dim1>, matrix_vector_product_fallback_type<dim0, dim1>>::f(matrix, input, output);
}

#elif defined(__SSE__)

template <size_t N>
struct dot_product_8_type {
  static constexpr size_t num_units = 2;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m128 sum_0 = _mm_setzero_ps();
    __m128 sum_1 = _mm_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      {
        const __m128 a_0 = _mm_load_ps(a + 0 * per_unit<float>);
        const __m128 b_0 = _mm_load_ps(b + 0 * per_unit<float>);
        sum_0 = _mm_add_ps(_mm_mul_ps(a_0, b_0), sum_0);
      }

      {
        const __m128 a_1 = _mm_load_ps(a + 1 * per_unit<float>);
        const __m128 b_1 = _mm_load_ps(b + 1 * per_unit<float>);
        sum_1 = _mm_add_ps(_mm_mul_ps(a_1, b_1), sum_1);
      }
    }

    const __m128 reduced_4 = _mm_add_ps(sum_0, sum_1);
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const float sum = _mm_cvtss_f32(reduced_1);
    return sum;
  }
};

template <size_t N>
struct dot_product_16_type {
  static constexpr size_t num_units = 4;
  static constexpr size_t per_iteration = per_unit<float> * num_units;
  static constexpr bool available = divides<N, per_iteration>;

  static inline float f(const float* a, const float* b) {
    __m128 sum_0 = _mm_setzero_ps();
    __m128 sum_1 = _mm_setzero_ps();
    __m128 sum_2 = _mm_setzero_ps();
    __m128 sum_3 = _mm_setzero_ps();

    for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
      {
        const __m128 a_0 = _mm_load_ps(a + 0 * per_unit<float>);
        const __m128 b_0 = _mm_load_ps(b + 0 * per_unit<float>);
        sum_0 = _mm_add_ps(_mm_mul_ps(a_0, b_0), sum_0);
      }

      {
        const __m128 a_1 = _mm_load_ps(a + 1 * per_unit<float>);
        const __m128 b_1 = _mm_load_ps(b + 1 * per_unit<float>);
        sum_1 = _mm_add_ps(_mm_mul_ps(a_1, b_1), sum_1);
      }

      {
        const __m128 a_2 = _mm_load_ps(a + 2 * per_unit<float>);
        const __m128 b_2 = _mm_load_ps(b + 2 * per_unit<float>);
        sum_2 = _mm_add_ps(_mm_mul_ps(a_2, b_2), sum_2);
      }

      {
        const __m128 a_3 = _mm_load_ps(a + 3 * per_unit<float>);
        const __m128 b_3 = _mm_load_ps(b + 3 * per_unit<float>);
        sum_3 = _mm_add_ps(_mm_mul_ps(a_3, b_3), sum_3);
      }
    }

    const __m128 reduced_4 = _mm_add_ps(_mm_add_ps(sum_0, sum_1), _mm_add_ps(sum_2, sum_3));
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const float sum = _mm_cvtss_f32(reduced_1);
    return sum;
  }
};

template <size_t N>
inline float dot_product(const float* a, const float* b) {
  return overload_set<dot_product_16_type<N>, dot_product_8_type<N> >::f(a, b);
}

#endif

}  // namespace simd
