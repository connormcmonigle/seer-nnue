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

#include <x86intrin.h>

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

#if defined(__AVX512BW__)
struct vector_512 {
  using integral_type = __m512i;
  using float_type = __m512;
  static_assert(sizeof(integral_type) == sizeof(float_type));
  static constexpr std::size_t size = sizeof(integral_type);
};
#endif

#if defined(__AVX2__)
struct vector_256 {
  using integral_type = __m256i;
  using float_type = __m256;
  static_assert(sizeof(integral_type) == sizeof(float_type));
  static constexpr std::size_t size = sizeof(integral_type);
};
#endif

#if defined(__SSSE3__)
struct vector_128 {
  using integral_type = __m128i;
  using float_type = __m128;
  static_assert(sizeof(integral_type) == sizeof(float_type));
  static constexpr std::size_t size = sizeof(integral_type);
};
#endif

#if defined(__AVX512BW__)
constexpr std::size_t alignment = vector_512::size;
#elif defined(__AVX2__)
constexpr std::size_t alignment = vector_256::size;
#elif defined(__SSSE3__)
constexpr std::size_t alignment = vector_128::size;
#else
constexpr std::size_t default_alignment = 16;
constexpr std::size_t alignment = default_alignment;
#endif

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

#if defined(__AVX512BW__)
template <std::size_t dim>
struct int16_add_x128 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_512, std::int16_t>>;

  static inline void f(std::int16_t* a, const std::int16_t* b) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_512, std::int16_t>) {
      __m512i* a_0 = (__m512i*)(a + i + 0 * per_unit<vector_512, std::int16_t>);
      *a_0 = _mm512_add_epi16(*a_0, _mm512_load_si512((__m512i*)(b + i + 0 * per_unit<vector_512, std::int16_t>)));

      __m512i* a_1 = (__m512i*)(a + i + 1 * per_unit<vector_512, std::int16_t>);
      *a_1 = _mm512_add_epi16(*a_1, _mm512_load_si512((__m512i*)(b + i + 1 * per_unit<vector_512, std::int16_t>)));

      __m512i* a_2 = (__m512i*)(a + i + 2 * per_unit<vector_512, std::int16_t>);
      *a_2 = _mm512_add_epi16(*a_2, _mm512_load_si512((__m512i*)(b + i + 2 * per_unit<vector_512, std::int16_t>)));

      __m512i* a_3 = (__m512i*)(a + i + 3 * per_unit<vector_512, std::int16_t>);
      *a_3 = _mm512_add_epi16(*a_3, _mm512_load_si512((__m512i*)(b + i + 3 * per_unit<vector_512, std::int16_t>)));
    }
  }
};

template <std::size_t dim>
inline void add(std::int16_t* a, const std::int16_t* b) noexcept {
  return overload_set<int16_add_x128<dim>>::f(a, b);
}

template <std::size_t dim>
struct int16_sub_x128 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_512, std::int16_t>>;

  static inline void f(std::int16_t* a, const std::int16_t* b) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_512, std::int16_t>) {
      __m512i* a_0 = (__m512i*)(a + i + 0 * per_unit<vector_512, std::int16_t>);
      *a_0 = _mm512_sub_epi16(*a_0, _mm512_load_si512((__m512i*)(b + i + 0 * per_unit<vector_512, std::int16_t>)));

      __m512i* a_1 = (__m512i*)(a + i + 1 * per_unit<vector_512, std::int16_t>);
      *a_1 = _mm512_sub_epi16(*a_1, _mm512_load_si512((__m512i*)(b + i + 1 * per_unit<vector_512, std::int16_t>)));

      __m512i* a_2 = (__m512i*)(a + i + 2 * per_unit<vector_512, std::int16_t>);
      *a_2 = _mm512_sub_epi16(*a_2, _mm512_load_si512((__m512i*)(b + i + 2 * per_unit<vector_512, std::int16_t>)));

      __m512i* a_3 = (__m512i*)(a + i + 3 * per_unit<vector_512, std::int16_t>);
      *a_3 = _mm512_sub_epi16(*a_3, _mm512_load_si512((__m512i*)(b + i + 3 * per_unit<vector_512, std::int16_t>)));
    }
  }
};

template <std::size_t dim>
inline void sub(std::int16_t* a, const std::int16_t* b) {
  return overload_set<int16_sub_x128<dim>>::f(a, b);
}

template <std::size_t dim>
struct int16_add_add_sub_x128 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_512, std::int16_t>>;

  static inline void f(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, std::int16_t* out) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_512, std::int16_t>) {
      {
        const __m512i a_0_0 = _mm512_load_si512((__m512i*)(a_0 + i + 0 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_0 = _mm512_load_si512((__m512i*)(a_1 + i + 0 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_0 = _mm512_load_si512((__m512i*)(s_0 + i + 0 * per_unit<vector_512, std::int16_t>));
        __m512i* out_0 = (__m512i*)(out + i + 0 * per_unit<vector_512, std::int16_t>);
        *out_0 = _mm512_add_epi16(a_0_0, _mm512_sub_epi16(a_1_0, s_0_0));
      }

      {
        const __m512i a_0_1 = _mm512_load_si512((__m512i*)(a_0 + i + 1 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_1 = _mm512_load_si512((__m512i*)(a_1 + i + 1 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_1 = _mm512_load_si512((__m512i*)(s_0 + i + 1 * per_unit<vector_512, std::int16_t>));
        __m512i* out_1 = (__m512i*)(out + i + 1 * per_unit<vector_512, std::int16_t>);
        *out_1 = _mm512_add_epi16(a_0_1, _mm512_sub_epi16(a_1_1, s_0_1));
      }

      {
        const __m512i a_0_2 = _mm512_load_si512((__m512i*)(a_0 + i + 2 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_2 = _mm512_load_si512((__m512i*)(a_1 + i + 2 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_2 = _mm512_load_si512((__m512i*)(s_0 + i + 2 * per_unit<vector_512, std::int16_t>));
        __m512i* out_2 = (__m512i*)(out + i + 2 * per_unit<vector_512, std::int16_t>);
        *out_2 = _mm512_add_epi16(a_0_2, _mm512_sub_epi16(a_1_2, s_0_2));
      }

      {
        const __m512i a_0_3 = _mm512_load_si512((__m512i*)(a_0 + i + 3 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_3 = _mm512_load_si512((__m512i*)(a_1 + i + 3 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_3 = _mm512_load_si512((__m512i*)(s_0 + i + 3 * per_unit<vector_512, std::int16_t>));
        __m512i* out_3 = (__m512i*)(out + i + 3 * per_unit<vector_512, std::int16_t>);
        *out_3 = _mm512_add_epi16(a_0_3, _mm512_sub_epi16(a_1_3, s_0_3));
      }
    }
  }
};

template <std::size_t dim>
inline void add_add_sub(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, std::int16_t* out) noexcept {
  return overload_set<int16_add_add_sub_x128<dim>>::f(a_0, a_1, s_0, out);
}

template <std::size_t dim>
struct int16_add_add_sub_sub_x128 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_512, std::int16_t>>;

  static inline void
  f(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, const std::int16_t* s_1, std::int16_t* out) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_512, std::int16_t>) {
      {
        const __m512i a_0_0 = _mm512_load_si512((__m512i*)(a_0 + i + 0 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_0 = _mm512_load_si512((__m512i*)(a_1 + i + 0 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_0 = _mm512_load_si512((__m512i*)(s_0 + i + 0 * per_unit<vector_512, std::int16_t>));
        const __m512i s_1_0 = _mm512_load_si512((__m512i*)(s_1 + i + 0 * per_unit<vector_512, std::int16_t>));
        __m512i* out_0 = (__m512i*)(out + i + 0 * per_unit<vector_512, std::int16_t>);
        *out_0 = _mm512_add_epi16(_mm512_sub_epi16(a_0_0, s_0_0), _mm512_sub_epi16(a_1_0, s_1_0));
      }

      {
        const __m512i a_0_1 = _mm512_load_si512((__m512i*)(a_0 + i + 1 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_1 = _mm512_load_si512((__m512i*)(a_1 + i + 1 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_1 = _mm512_load_si512((__m512i*)(s_0 + i + 1 * per_unit<vector_512, std::int16_t>));
        const __m512i s_1_1 = _mm512_load_si512((__m512i*)(s_1 + i + 1 * per_unit<vector_512, std::int16_t>));
        __m512i* out_1 = (__m512i*)(out + i + 1 * per_unit<vector_512, std::int16_t>);
        *out_1 = _mm512_add_epi16(_mm512_sub_epi16(a_0_1, s_0_1), _mm512_sub_epi16(a_1_1, s_1_1));
      }

      {
        const __m512i a_0_2 = _mm512_load_si512((__m512i*)(a_0 + i + 2 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_2 = _mm512_load_si512((__m512i*)(a_1 + i + 2 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_2 = _mm512_load_si512((__m512i*)(s_0 + i + 2 * per_unit<vector_512, std::int16_t>));
        const __m512i s_1_2 = _mm512_load_si512((__m512i*)(s_1 + i + 2 * per_unit<vector_512, std::int16_t>));
        __m512i* out_2 = (__m512i*)(out + i + 2 * per_unit<vector_512, std::int16_t>);
        *out_2 = _mm512_add_epi16(_mm512_sub_epi16(a_0_2, s_0_2), _mm512_sub_epi16(a_1_2, s_1_2));
      }

      {
        const __m512i a_0_3 = _mm512_load_si512((__m512i*)(a_0 + i + 3 * per_unit<vector_512, std::int16_t>));
        const __m512i a_1_3 = _mm512_load_si512((__m512i*)(a_1 + i + 3 * per_unit<vector_512, std::int16_t>));
        const __m512i s_0_3 = _mm512_load_si512((__m512i*)(s_0 + i + 3 * per_unit<vector_512, std::int16_t>));
        const __m512i s_1_3 = _mm512_load_si512((__m512i*)(s_1 + i + 3 * per_unit<vector_512, std::int16_t>));
        __m512i* out_3 = (__m512i*)(out + i + 3 * per_unit<vector_512, std::int16_t>);
        *out_3 = _mm512_add_epi16(_mm512_sub_epi16(a_0_3, s_0_3), _mm512_sub_epi16(a_1_3, s_1_3));
      }
    }
  }
};

template <std::size_t dim>
inline void
add_add_sub_sub(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, const std::int16_t* s_1, std::int16_t* out) noexcept {
  return overload_set<int16_add_add_sub_sub_x128<dim>>::f(a_0, a_1, s_0, s_1, out);
}

template <std::size_t dim0, std::size_t dim1>
struct float_relu_matrix_vector_product_x8_x1 {
  static constexpr bool available = divides<dim0, per_unit<vector_256, float>>;

  static inline void f(const float* matrix, const float* input, float* output) {
    const __m256 zero = _mm256_setzero_ps();
    for (std::size_t i(0); i < dim1; ++i) {
      __m256 sum = _mm256_setzero_ps();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_256, float>) {
        const __m256 input_region = _mm256_max_ps(zero, _mm256_load_ps(input + j));
        sum = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + i * dim0 + j), input_region), sum);
      }

      const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 0x1));
      const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
      const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));

      output[i] += _mm_cvtss_f32(reduced_1);
    }
  }
};

template <std::size_t dim0, std::size_t dim1>
struct float_relu_matrix_vector_product_x8_x8 {
  static constexpr std::size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<vector_256, float>>;

  static inline void f(const float* matrix, const float* input, float* output) noexcept {
    const __m256 zero = _mm256_setzero_ps();
    __m256* v_output = (__m256*)output;
    constexpr std::size_t output_step = num_units / per_unit<vector_256, float>;
    for (std::size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m256 sum_0 = _mm256_setzero_ps();
      __m256 sum_1 = _mm256_setzero_ps();
      __m256 sum_2 = _mm256_setzero_ps();
      __m256 sum_3 = _mm256_setzero_ps();
      __m256 sum_4 = _mm256_setzero_ps();
      __m256 sum_5 = _mm256_setzero_ps();
      __m256 sum_6 = _mm256_setzero_ps();
      __m256 sum_7 = _mm256_setzero_ps();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_256, float>) {
        const __m256 input_region = _mm256_max_ps(zero, _mm256_load_ps(input + j));
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

template <std::size_t dim0, std::size_t dim1>
struct int16_relu_matrix_vector_product_x32_x8 {
  static constexpr std::size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<vector_512, std::int16_t>>;

  static inline void f(const std::int16_t* matrix, const std::int16_t* input, std::int32_t* output) noexcept {
    const __m512i zero = _mm512_setzero_si512();

    const __m512i mm512_unpacklo_epi128_permutationx2var =
        _mm512_set_epi32(0x17, 0x16, 0x15, 0x14, 0x07, 0x06, 0x05, 0x04, 0x13, 0x12, 0x11, 0x10, 0x03, 0x02, 0x01, 0x00);

    const __m512i mm512_unpackhi_epi128_permutationx2var =
        _mm512_set_epi32(0x1f, 0x1e, 0x1d, 0x1c, 0x0f, 0x0e, 0x0d, 0x0c, 0x1b, 0x1a, 0x19, 0x18, 0x0b, 0x0a, 0x09, 0x08);

    __m256i* v_output = (__m256i*)output;
    constexpr std::size_t output_step = num_units / per_unit<vector_256, std::int32_t>;
    for (std::size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m512i sum_0 = _mm512_setzero_si512();
      __m512i sum_1 = _mm512_setzero_si512();
      __m512i sum_2 = _mm512_setzero_si512();
      __m512i sum_3 = _mm512_setzero_si512();
      __m512i sum_4 = _mm512_setzero_si512();
      __m512i sum_5 = _mm512_setzero_si512();
      __m512i sum_6 = _mm512_setzero_si512();
      __m512i sum_7 = _mm512_setzero_si512();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_512, std::int16_t>) {
        const __m512i input_region = _mm512_max_epi16(zero, _mm512_load_si512((__m512i*)(input + j)));
        sum_0 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 0) * dim0 + j)), input_region), sum_0);
        sum_1 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 1) * dim0 + j)), input_region), sum_1);
        sum_2 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 2) * dim0 + j)), input_region), sum_2);
        sum_3 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 3) * dim0 + j)), input_region), sum_3);
        sum_4 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 4) * dim0 + j)), input_region), sum_4);
        sum_5 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 5) * dim0 + j)), input_region), sum_5);
        sum_6 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 6) * dim0 + j)), input_region), sum_6);
        sum_7 = _mm512_add_epi32(_mm512_madd_epi16(_mm512_load_si512((__m512i*)(matrix + (i + 7) * dim0 + j)), input_region), sum_7);
      }

      const __m512i sum_01 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum_0, sum_1), _mm512_unpackhi_epi32(sum_0, sum_1));
      const __m512i sum_23 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum_2, sum_3), _mm512_unpackhi_epi32(sum_2, sum_3));
      const __m512i sum_45 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum_4, sum_5), _mm512_unpackhi_epi32(sum_4, sum_5));
      const __m512i sum_67 = _mm512_add_epi32(_mm512_unpacklo_epi32(sum_6, sum_7), _mm512_unpackhi_epi32(sum_6, sum_7));

      const __m512i sum_0123 = _mm512_add_epi32(_mm512_unpacklo_epi64(sum_01, sum_23), _mm512_unpackhi_epi64(sum_01, sum_23));
      const __m512i sum_4567 = _mm512_add_epi32(_mm512_unpacklo_epi64(sum_45, sum_67), _mm512_unpackhi_epi64(sum_45, sum_67));

      const __m512i sum_512_01234567 = _mm512_add_epi32(
          _mm512_permutex2var_epi32(sum_0123, mm512_unpacklo_epi128_permutationx2var, sum_4567),
          _mm512_permutex2var_epi32(sum_0123, mm512_unpackhi_epi128_permutationx2var, sum_4567));

      const __m256i sum_256_01234567 = _mm256_add_epi32(_mm512_castsi512_si256(sum_512_01234567), _mm512_extracti64x4_epi64(sum_512_01234567, 0x1));

      *v_output = _mm256_add_epi32(*v_output, sum_256_01234567);
    }
  }
};

template <std::size_t dim0, std::size_t dim1>
inline void relu_matrix_vector_product(const float* matrix, const float* input, float* output) noexcept {
  return overload_set<float_relu_matrix_vector_product_x8_x8<dim0, dim1>, float_relu_matrix_vector_product_x8_x1<dim0, dim1>>::f(
      matrix, input, output);
}

template <std::size_t dim0, std::size_t dim1>
inline void relu_matrix_vector_product(const std::int16_t* matrix, const std::int16_t* input, std::int32_t* output) noexcept {
  return overload_set<int16_relu_matrix_vector_product_x32_x8<dim0, dim1>>::f(matrix, input, output);
}

#elif defined(__AVX2__)
template <std::size_t dim>
struct int16_add_x64 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_256, std::int16_t>>;

  static inline void f(std::int16_t* a, const std::int16_t* b) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_256, std::int16_t>) {
      __m256i* a_0 = (__m256i*)(a + i + 0 * per_unit<vector_256, std::int16_t>);
      *a_0 = _mm256_add_epi16(*a_0, _mm256_load_si256((__m256i*)(b + i + 0 * per_unit<vector_256, std::int16_t>)));

      __m256i* a_1 = (__m256i*)(a + i + 1 * per_unit<vector_256, std::int16_t>);
      *a_1 = _mm256_add_epi16(*a_1, _mm256_load_si256((__m256i*)(b + i + 1 * per_unit<vector_256, std::int16_t>)));

      __m256i* a_2 = (__m256i*)(a + i + 2 * per_unit<vector_256, std::int16_t>);
      *a_2 = _mm256_add_epi16(*a_2, _mm256_load_si256((__m256i*)(b + i + 2 * per_unit<vector_256, std::int16_t>)));

      __m256i* a_3 = (__m256i*)(a + i + 3 * per_unit<vector_256, std::int16_t>);
      *a_3 = _mm256_add_epi16(*a_3, _mm256_load_si256((__m256i*)(b + i + 3 * per_unit<vector_256, std::int16_t>)));
    }
  }
};

template <std::size_t dim>
inline void add(std::int16_t* a, const std::int16_t* b) noexcept {
  return overload_set<int16_add_x64<dim>>::f(a, b);
}

template <std::size_t dim>
struct int16_sub_x64 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_256, std::int16_t>>;

  static inline void f(std::int16_t* a, const std::int16_t* b) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_256, std::int16_t>) {
      __m256i* a_0 = (__m256i*)(a + i + 0 * per_unit<vector_256, std::int16_t>);
      *a_0 = _mm256_sub_epi16(*a_0, _mm256_load_si256((__m256i*)(b + i + 0 * per_unit<vector_256, std::int16_t>)));

      __m256i* a_1 = (__m256i*)(a + i + 1 * per_unit<vector_256, std::int16_t>);
      *a_1 = _mm256_sub_epi16(*a_1, _mm256_load_si256((__m256i*)(b + i + 1 * per_unit<vector_256, std::int16_t>)));

      __m256i* a_2 = (__m256i*)(a + i + 2 * per_unit<vector_256, std::int16_t>);
      *a_2 = _mm256_sub_epi16(*a_2, _mm256_load_si256((__m256i*)(b + i + 2 * per_unit<vector_256, std::int16_t>)));

      __m256i* a_3 = (__m256i*)(a + i + 3 * per_unit<vector_256, std::int16_t>);
      *a_3 = _mm256_sub_epi16(*a_3, _mm256_load_si256((__m256i*)(b + i + 3 * per_unit<vector_256, std::int16_t>)));
    }
  }
};

template <std::size_t dim>
inline void sub(std::int16_t* a, const std::int16_t* b) noexcept {
  return overload_set<int16_sub_x64<dim>>::f(a, b);
}

template <std::size_t dim>
struct int16_add_add_sub_x64 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_256, std::int16_t>>;

  static inline void f(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, std::int16_t* out) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_256, std::int16_t>) {
      {
        const __m256i a_0_0 = _mm256_load_si256((__m256i*)(a_0 + i + 0 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_0 = _mm256_load_si256((__m256i*)(a_1 + i + 0 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_0 = _mm256_load_si256((__m256i*)(s_0 + i + 0 * per_unit<vector_256, std::int16_t>));
        __m256i* out_0 = (__m256i*)(out + i + 0 * per_unit<vector_256, std::int16_t>);
        *out_0 = _mm256_add_epi16(a_0_0, _mm256_sub_epi16(a_1_0, s_0_0));
      }

      {
        const __m256i a_0_1 = _mm256_load_si256((__m256i*)(a_0 + i + 1 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_1 = _mm256_load_si256((__m256i*)(a_1 + i + 1 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_1 = _mm256_load_si256((__m256i*)(s_0 + i + 1 * per_unit<vector_256, std::int16_t>));
        __m256i* out_1 = (__m256i*)(out + i + 1 * per_unit<vector_256, std::int16_t>);
        *out_1 = _mm256_add_epi16(a_0_1, _mm256_sub_epi16(a_1_1, s_0_1));
      }

      {
        const __m256i a_0_2 = _mm256_load_si256((__m256i*)(a_0 + i + 2 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_2 = _mm256_load_si256((__m256i*)(a_1 + i + 2 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_2 = _mm256_load_si256((__m256i*)(s_0 + i + 2 * per_unit<vector_256, std::int16_t>));
        __m256i* out_2 = (__m256i*)(out + i + 2 * per_unit<vector_256, std::int16_t>);
        *out_2 = _mm256_add_epi16(a_0_2, _mm256_sub_epi16(a_1_2, s_0_2));
      }

      {
        const __m256i a_0_3 = _mm256_load_si256((__m256i*)(a_0 + i + 3 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_3 = _mm256_load_si256((__m256i*)(a_1 + i + 3 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_3 = _mm256_load_si256((__m256i*)(s_0 + i + 3 * per_unit<vector_256, std::int16_t>));
        __m256i* out_3 = (__m256i*)(out + i + 3 * per_unit<vector_256, std::int16_t>);
        *out_3 = _mm256_add_epi16(a_0_3, _mm256_sub_epi16(a_1_3, s_0_3));
      }
    }
  }
};

template <std::size_t dim>
inline void add_add_sub(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, std::int16_t* out) {
  return overload_set<int16_add_add_sub_x64<dim>>::f(a_0, a_1, s_0, out);
}

template <std::size_t dim>
struct int16_add_add_sub_sub_x64 {
  static constexpr std::size_t num_units = 4;
  static constexpr bool available = divides<dim, num_units * per_unit<vector_256, std::int16_t>>;

  static inline void
  f(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, const std::int16_t* s_1, std::int16_t* out) noexcept {
    for (std::size_t i(0); i < dim; i += num_units * per_unit<vector_256, std::int16_t>) {
      {
        const __m256i a_0_0 = _mm256_load_si256((__m256i*)(a_0 + i + 0 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_0 = _mm256_load_si256((__m256i*)(a_1 + i + 0 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_0 = _mm256_load_si256((__m256i*)(s_0 + i + 0 * per_unit<vector_256, std::int16_t>));
        const __m256i s_1_0 = _mm256_load_si256((__m256i*)(s_1 + i + 0 * per_unit<vector_256, std::int16_t>));
        __m256i* out_0 = (__m256i*)(out + i + 0 * per_unit<vector_256, std::int16_t>);
        *out_0 = _mm256_add_epi16(_mm256_sub_epi16(a_0_0, s_0_0), _mm256_sub_epi16(a_1_0, s_1_0));
      }

      {
        const __m256i a_0_1 = _mm256_load_si256((__m256i*)(a_0 + i + 1 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_1 = _mm256_load_si256((__m256i*)(a_1 + i + 1 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_1 = _mm256_load_si256((__m256i*)(s_0 + i + 1 * per_unit<vector_256, std::int16_t>));
        const __m256i s_1_1 = _mm256_load_si256((__m256i*)(s_1 + i + 1 * per_unit<vector_256, std::int16_t>));
        __m256i* out_1 = (__m256i*)(out + i + 1 * per_unit<vector_256, std::int16_t>);
        *out_1 = _mm256_add_epi16(_mm256_sub_epi16(a_0_1, s_0_1), _mm256_sub_epi16(a_1_1, s_1_1));
      }

      {
        const __m256i a_0_2 = _mm256_load_si256((__m256i*)(a_0 + i + 2 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_2 = _mm256_load_si256((__m256i*)(a_1 + i + 2 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_2 = _mm256_load_si256((__m256i*)(s_0 + i + 2 * per_unit<vector_256, std::int16_t>));
        const __m256i s_1_2 = _mm256_load_si256((__m256i*)(s_1 + i + 2 * per_unit<vector_256, std::int16_t>));
        __m256i* out_2 = (__m256i*)(out + i + 2 * per_unit<vector_256, std::int16_t>);
        *out_2 = _mm256_add_epi16(_mm256_sub_epi16(a_0_2, s_0_2), _mm256_sub_epi16(a_1_2, s_1_2));
      }

      {
        const __m256i a_0_3 = _mm256_load_si256((__m256i*)(a_0 + i + 3 * per_unit<vector_256, std::int16_t>));
        const __m256i a_1_3 = _mm256_load_si256((__m256i*)(a_1 + i + 3 * per_unit<vector_256, std::int16_t>));
        const __m256i s_0_3 = _mm256_load_si256((__m256i*)(s_0 + i + 3 * per_unit<vector_256, std::int16_t>));
        const __m256i s_1_3 = _mm256_load_si256((__m256i*)(s_1 + i + 3 * per_unit<vector_256, std::int16_t>));
        __m256i* out_3 = (__m256i*)(out + i + 3 * per_unit<vector_256, std::int16_t>);
        *out_3 = _mm256_add_epi16(_mm256_sub_epi16(a_0_3, s_0_3), _mm256_sub_epi16(a_1_3, s_1_3));
      }
    }
  }
};

template <std::size_t dim>
inline void
add_add_sub_sub(const std::int16_t* a_0, const std::int16_t* a_1, const std::int16_t* s_0, const std::int16_t* s_1, std::int16_t* out) noexcept {
  return overload_set<int16_add_add_sub_sub_x64<dim>>::f(a_0, a_1, s_0, s_1, out);
}

template <std::size_t dim0, std::size_t dim1>
struct float_relu_matrix_vector_product_x8_x1 {
  static constexpr bool available = divides<dim0, per_unit<vector_256, float>>;

  static inline void f(const float* matrix, const float* input, float* output) {
    const __m256 zero = _mm256_setzero_ps();
    for (std::size_t i(0); i < dim1; ++i) {
      __m256 sum = _mm256_setzero_ps();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_256, float>) {
        const __m256 input_region = _mm256_max_ps(zero, _mm256_load_ps(input + j));
        sum = _mm256_add_ps(_mm256_mul_ps(_mm256_load_ps(matrix + i * dim0 + j), input_region), sum);
      }

      const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(sum), _mm256_extractf128_ps(sum, 0x1));
      const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
      const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));

      output[i] += _mm_cvtss_f32(reduced_1);
    }
  }
};

template <std::size_t dim0, std::size_t dim1>
struct float_relu_matrix_vector_product_x8_x8 {
  static constexpr std::size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<vector_256, float>>;

  static inline void f(const float* matrix, const float* input, float* output) noexcept {
    const __m256 zero = _mm256_setzero_ps();
    __m256* v_output = (__m256*)output;
    constexpr std::size_t output_step = num_units / per_unit<vector_256, float>;
    for (std::size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m256 sum_0 = _mm256_setzero_ps();
      __m256 sum_1 = _mm256_setzero_ps();
      __m256 sum_2 = _mm256_setzero_ps();
      __m256 sum_3 = _mm256_setzero_ps();
      __m256 sum_4 = _mm256_setzero_ps();
      __m256 sum_5 = _mm256_setzero_ps();
      __m256 sum_6 = _mm256_setzero_ps();
      __m256 sum_7 = _mm256_setzero_ps();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_256, float>) {
        const __m256 input_region = _mm256_max_ps(zero, _mm256_load_ps(input + j));
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

template <std::size_t dim0, std::size_t dim1>
struct int16_sqcrelu512_matrix_vector_product_x16_x8 {
  static constexpr std::size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<vector_256, std::int16_t>>;

  static inline void f(const std::int16_t* matrix, const std::int16_t* input, std::int32_t* output) noexcept {
    const __m256i fill_zero = _mm256_setzero_si256();
    const __m256i fill_512 = _mm256_set_epi16(512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512, 512);

    __m256i* v_output = (__m256i*)output;
    constexpr std::size_t output_step = num_units / per_unit<vector_256, std::int32_t>;
    for (std::size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m256i sum_0 = _mm256_setzero_si256();
      __m256i sum_1 = _mm256_setzero_si256();
      __m256i sum_2 = _mm256_setzero_si256();
      __m256i sum_3 = _mm256_setzero_si256();
      __m256i sum_4 = _mm256_setzero_si256();
      __m256i sum_5 = _mm256_setzero_si256();
      __m256i sum_6 = _mm256_setzero_si256();
      __m256i sum_7 = _mm256_setzero_si256();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_256, std::int16_t>) {
        constexpr int srli_pre = 3;
        constexpr int srli_post = 6;

        __m256i input_region;
        input_region = _mm256_min_epi16(fill_512, _mm256_max_epi16(fill_zero, _mm256_load_si256((__m256i*)(input + j))));
        input_region = _mm256_srli_epi16(_mm256_mullo_epi16(input_region, _mm256_srli_epi16(input_region, srli_pre)), srli_post);

        sum_0 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 0) * dim0 + j)), input_region), sum_0);
        sum_1 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 1) * dim0 + j)), input_region), sum_1);
        sum_2 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 2) * dim0 + j)), input_region), sum_2);
        sum_3 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 3) * dim0 + j)), input_region), sum_3);
        sum_4 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 4) * dim0 + j)), input_region), sum_4);
        sum_5 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 5) * dim0 + j)), input_region), sum_5);
        sum_6 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 6) * dim0 + j)), input_region), sum_6);
        sum_7 = _mm256_add_epi32(_mm256_madd_epi16(_mm256_load_si256((__m256i*)(matrix + (i + 7) * dim0 + j)), input_region), sum_7);
      }

      const __m256i sum_01 = _mm256_hadd_epi32(sum_0, sum_1);
      const __m256i sum_23 = _mm256_hadd_epi32(sum_2, sum_3);
      const __m256i sum_45 = _mm256_hadd_epi32(sum_4, sum_5);
      const __m256i sum_67 = _mm256_hadd_epi32(sum_6, sum_7);

      const __m256i sum_0123 = _mm256_hadd_epi32(sum_01, sum_23);
      const __m256i sum_4567 = _mm256_hadd_epi32(sum_45, sum_67);

      const __m256i sum_01234567 =
          _mm256_add_epi32(_mm256_permute2f128_si256(sum_0123, sum_4567, 0x20), _mm256_permute2f128_si256(sum_0123, sum_4567, 0x31));

      *v_output = _mm256_add_epi32(*v_output, sum_01234567);
    }
  }
};

template <std::size_t dim0, std::size_t dim1>
inline void relu_matrix_vector_product(const float* matrix, const float* input, float* output) noexcept {
  return overload_set<float_relu_matrix_vector_product_x8_x8<dim0, dim1>, float_relu_matrix_vector_product_x8_x1<dim0, dim1>>::f(
      matrix, input, output);
}

template <std::size_t dim0, std::size_t dim1>
inline void sqcrelu512_matrix_vector_product(const std::int16_t* matrix, const std::int16_t* input, std::int32_t* output) noexcept {
  return overload_set<int16_sqcrelu512_matrix_vector_product_x16_x8<dim0, dim1>>::f(matrix, input, output);
}

#elif defined(__SSSE3__)
template <std::size_t dim0, std::size_t dim1>
struct float_relu_matrix_vector_product_x8_x1 {
  static constexpr std::size_t num_units = 2;
  static constexpr std::size_t per_iteration = per_unit<vector_128, float> * num_units;
  static constexpr bool available = divides<dim0, per_iteration>;

  static inline void f(const float* matrix, const float* input, float* output) noexcept {
    const __m128 zero = _mm_setzero_ps();
    for (std::size_t i(0); i < dim1; ++i) {
      __m128 sum_0 = _mm_setzero_ps();
      __m128 sum_1 = _mm_setzero_ps();

      for (std::size_t j(0); j < dim0; j += per_iteration) {
        const __m128 input_region_0 = _mm_max_ps(zero, _mm_load_ps(input + j + 0 * per_unit<vector_128, float>));
        const __m128 input_region_1 = _mm_max_ps(zero, _mm_load_ps(input + j + 1 * per_unit<vector_128, float>));

        sum_0 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + i * dim0 + j + 0 * per_unit<vector_128, float>), input_region_0), sum_0);
        sum_1 = _mm_add_ps(_mm_mul_ps(_mm_load_ps(matrix + i * dim0 + j + 1 * per_unit<vector_128, float>), input_region_1), sum_1);
      }

      const __m128 reduced_4 = _mm_add_ps(sum_0, sum_1);
      const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
      const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
      output[i] += _mm_cvtss_f32(reduced_1);
    }
  }
};

template <std::size_t dim0, std::size_t dim1>
struct float_relu_matrix_vector_product_x4_x8 {
  static constexpr std::size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<vector_128, float>>;

  static inline void f(const float* matrix, const float* input, float* output) noexcept {
    const __m128 zero = _mm_setzero_ps();
    __m128* v_output = (__m128*)output;
    constexpr std::size_t output_step = num_units / per_unit<vector_128, float>;
    for (std::size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m128 sum_0 = _mm_setzero_ps();
      __m128 sum_1 = _mm_setzero_ps();
      __m128 sum_2 = _mm_setzero_ps();
      __m128 sum_3 = _mm_setzero_ps();
      __m128 sum_4 = _mm_setzero_ps();
      __m128 sum_5 = _mm_setzero_ps();
      __m128 sum_6 = _mm_setzero_ps();
      __m128 sum_7 = _mm_setzero_ps();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_128, float>) {
        const __m128 input_region = _mm_max_ps(zero, _mm_load_ps(input + j));
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

template <std::size_t dim0, std::size_t dim1>
struct int16_relu_matrix_vector_product_x8_x8 {
  static constexpr std::size_t num_units = 8;
  static constexpr bool available = divides<dim1, num_units> && divides<dim0, per_unit<vector_128, std::int16_t>>;

  static inline void f(const std::int16_t* matrix, const std::int16_t* input, std::int32_t* output) noexcept {
    const __m128i zero = _mm_setzero_si128();
    __m128i* v_output = (__m128i*)output;
    constexpr std::size_t output_step = num_units / per_unit<vector_128, std::int32_t>;
    for (std::size_t i(0); i < dim1; i += num_units, v_output += output_step) {
      __m128i sum_0 = _mm_setzero_si128();
      __m128i sum_1 = _mm_setzero_si128();
      __m128i sum_2 = _mm_setzero_si128();
      __m128i sum_3 = _mm_setzero_si128();
      __m128i sum_4 = _mm_setzero_si128();
      __m128i sum_5 = _mm_setzero_si128();
      __m128i sum_6 = _mm_setzero_si128();
      __m128i sum_7 = _mm_setzero_si128();

      for (std::size_t j(0); j < dim0; j += per_unit<vector_128, std::int16_t>) {
        const __m128i input_region = _mm_max_epi16(zero, _mm_load_si128((__m128i*)(input + j)));
        sum_0 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 0) * dim0 + j)), input_region), sum_0);
        sum_1 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 1) * dim0 + j)), input_region), sum_1);
        sum_2 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 2) * dim0 + j)), input_region), sum_2);
        sum_3 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 3) * dim0 + j)), input_region), sum_3);
        sum_4 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 4) * dim0 + j)), input_region), sum_4);
        sum_5 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 5) * dim0 + j)), input_region), sum_5);
        sum_6 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 6) * dim0 + j)), input_region), sum_6);
        sum_7 = _mm_add_epi32(_mm_madd_epi16(_mm_load_si128((__m128i*)(matrix + (i + 7) * dim0 + j)), input_region), sum_7);
      }

      const __m128i sum_01 = _mm_hadd_epi32(sum_0, sum_1);
      const __m128i sum_23 = _mm_hadd_epi32(sum_2, sum_3);
      const __m128i sum_45 = _mm_hadd_epi32(sum_4, sum_5);
      const __m128i sum_67 = _mm_hadd_epi32(sum_6, sum_7);

      const __m128i sum_0123 = _mm_hadd_epi32(sum_01, sum_23);
      const __m128i sum_4567 = _mm_hadd_epi32(sum_45, sum_67);

      *(v_output + 0) = _mm_add_epi32(*(v_output + 0), sum_0123);
      *(v_output + 1) = _mm_add_epi32(*(v_output + 1), sum_4567);
    }
  }
};

template <std::size_t dim0, std::size_t dim1>
inline void relu_matrix_vector_product(const float* matrix, const float* input, float* output) noexcept {
  return overload_set<float_relu_matrix_vector_product_x4_x8<dim0, dim1>, float_relu_matrix_vector_product_x8_x1<dim0, dim1>>::f(
      matrix, input, output);
}

template <std::size_t dim0, std::size_t dim1>
inline void relu_matrix_vector_product(const std::int16_t* matrix, const std::int16_t* input, std::int32_t* output) noexcept {
  return overload_set<int16_relu_matrix_vector_product_x8_x8<dim0, dim1>>::f(matrix, input, output);
}

#endif

}  // namespace simd
