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

#include <dot_type.h>
#include <immintrin.h>

#include <cstdint>
#include <type_traits>

namespace simd {

constexpr size_t alignment = 32;

template <size_t N, typename T>
inline util::dot_type<T> dot_product(const T* a, const T* b) {
  util::dot_type<T> sum{};
#pragma omp simd
  for (size_t i = 0; i < N; ++i) { sum += static_cast<util::dot_type<T>>(a[i] * b[i]); }
  return sum;
}

template <size_t N>
inline std::int32_t dot_product(const std::int16_t* a, const std::int16_t* b) {
  constexpr size_t num_units = 4;
  constexpr size_t per_unit = alignment / sizeof(std::int16_t);
  constexpr size_t per_iteration = per_unit * num_units;
  static_assert(N % per_iteration == 0, "N must be divisible by per_iteration");

  const __m256i* a_ = (const __m256i*)a;
  const __m256i* b_ = (const __m256i*)b;

  __m256i sum_0 = _mm256_setzero_si256();
  __m256i sum_1 = _mm256_setzero_si256();
  __m256i sum_2 = _mm256_setzero_si256();
  __m256i sum_3 = _mm256_setzero_si256();

  for (size_t i = 0; i < N; i += per_iteration, a_ += num_units, b_ += num_units) {
    sum_0 = _mm256_add_epi32(sum_0, _mm256_madd_epi16(a_[0], b_[0]));
    sum_1 = _mm256_add_epi32(sum_1, _mm256_madd_epi16(a_[1], b_[1]));
    sum_2 = _mm256_add_epi32(sum_2, _mm256_madd_epi16(a_[2], b_[2]));
    sum_3 = _mm256_add_epi32(sum_3, _mm256_madd_epi16(a_[3], b_[3]));
  }

  const __m256i reduced_8 = _mm256_add_epi32(_mm256_add_epi32(sum_0, sum_1), _mm256_add_epi32(sum_2, sum_3));
  const __m128i reduced_4 = _mm_add_epi32(_mm256_castsi256_si128(reduced_8), _mm256_extractf128_si256(reduced_8, 1));
  const __m128i reduced_2 = _mm_add_epi32(reduced_4, _mm_srli_si128(reduced_4, 8));
  const __m128i reduced_1 = _mm_add_epi32(reduced_2, _mm_srli_si128(reduced_2, 4));
  const std::int32_t sum = _mm_cvtsi128_si32(reduced_1);
  return sum;
}

template <size_t N>
inline float dot_product(const float* a, const float* b) {
  constexpr size_t num_units = 2;
  constexpr size_t per_unit = alignment / sizeof(float);
  constexpr size_t per_iteration = per_unit * num_units;
  static_assert(N % per_iteration == 0, "N must be divisible by per_iteration");

  const __m256* a_ = (const __m256*)a;
  const __m256* b_ = (const __m256*)b;

  __m256 sum_0 = _mm256_setzero_ps();
  __m256 sum_1 = _mm256_setzero_ps();

  for (size_t i = 0; i < N; i += per_iteration, a_ += num_units, b_ += num_units) {
    sum_0 = _mm256_add_ps(sum_0, _mm256_mul_ps(a_[0], b_[0]));
    sum_1 = _mm256_add_ps(sum_1, _mm256_mul_ps(a_[1], b_[1]));
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

template <size_t N, typename T, typename U>
inline void mul_convert(const T* in, U* out, const U& mul) {
  T sum{};
#pragma omp simd
  for (size_t i = 0; i < N; ++i) { out[i] = static_cast<U>(in[i]) * mul; }
  return sum;
}

template <size_t N>
inline void mul_convert(const std::int32_t* in, float* out, const float& mul) {
  constexpr size_t per_unit = alignment / sizeof(float);
  constexpr size_t per_iteration = per_unit;

  const __m256 mul_ = _mm256_set1_ps(mul);
  const __m256i* in_ = (const __m256i*)in;
  __m256* out_ = (__m256*)out;

  for (size_t i = 0; i < N; i += per_iteration, ++in_, ++out_) { *out_ = _mm256_mul_ps(_mm256_cvtepi32_ps(*in_), mul_); }
}

}  // namespace simd