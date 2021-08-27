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

template<typename T>
inline constexpr size_t per_unit = alignment / sizeof(T);

template <size_t N, typename T>
inline T dot_product(const T* a, const T* b) {
  T sum{};
#pragma omp simd
  for (size_t i = 0; i < N; ++i) { sum += a[i] * b[i]; }
  return sum;
}

#if defined(__AVX512DQ__)
template <size_t N>
inline float dot_product(const float* a, const float* b) {
  constexpr size_t num_units = 1;
  constexpr size_t per_iteration = per_unit<float> * num_units;
  static_assert(N % per_iteration == 0, "N must be divisible by per_iteration");
  __m512 sum_ = _mm512_setzero_ps();

  for (size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration) {
    const __m512 a_ = _mm512_load_ps(a);
    const __m512 b_ = _mm512_load_ps(b);
    sum_ = _mm512_fmadd_ps(a_, b_, sum_);
  }

  // avoids extra move instruction by casting sum, adds lower 8 float elements to upper 8 float elements
  const __m256 reduced_8 = _mm256_add_ps(_mm512_castps512_ps256(sum_), _mm512_extractf32x8_ps(sum_, 1));
  // avoids extra move instruction by casting sum, adds lower 4 float elements to upper 4 float elements
  const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
  // adds lower 2 float elements to the upper 2
  const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
  // adds 0th float element to 1st float element
  const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 1));
  const float sum = _mm_cvtss_f32(reduced_1);
  return sum;
}

#elif defined(__AVX__)
template <size_t N>
inline float dot_product(const float* a, const float* b) {
  constexpr size_t num_units = 2;
  constexpr size_t per_iteration = per_unit<float> * num_units;
  static_assert(N % per_iteration == 0, "N must be divisible by per_iteration");
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
  // avoids extra move instruction by casting sum, adds lower 4 float elements to upper 4 float elements
  const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
  // adds lower 2 float elements to the upper 2
  const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
  // adds 0th float element to 1st float element
  const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
  const float sum = _mm_cvtss_f32(reduced_1);
  return sum;
}

#elif defined(__SSE__)
template <size_t N>
inline float dot_product(const float* a, const float* b) {
  constexpr size_t num_units = 4;
  constexpr size_t per_iteration = per_unit<float> * num_units;
  static_assert(N % per_iteration == 0, "N must be divisible by per_iteration");
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
  // adds lower 2 float elements to the upper 2
  const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
  // adds 0th float element to 1st float element
  const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
  const float sum = _mm_cvtss_f32(reduced_1);
  return sum;
}
#endif

}  // namespace simd
