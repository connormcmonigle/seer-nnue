#pragma once
#include <type_traits>

#ifdef __AVX2__
#include <immintrin.h>
#endif


namespace simd{

constexpr size_t alignment = 32;

template<typename T, size_t N>
inline T dot_product_(const T* a, const T* b){
  T sum{};
#pragma omp simd
  for(size_t i = 0; i < N; ++i){
    sum += a[i] * b[i];
  }
  return sum;
}

template<typename T, size_t N>
inline T dot_product(const T* a, const T* b){
#ifdef __AVX2__
  if constexpr(std::is_same_v<T, float>){
    static_assert(alignment % sizeof(T) == 0, "alignment must be divisible by sizeof(T)");
    constexpr size_t num_units = 4;
    constexpr size_t per_unit = alignment / sizeof(T);
    constexpr size_t per_iteration = per_unit * num_units;
    static_assert(N % per_iteration == 0, "N must be divisible by per_iteration");
    __m256 sum_0 = _mm256_setzero_ps();
    __m256 sum_1 = _mm256_setzero_ps();
    __m256 sum_2 = _mm256_setzero_ps();
    __m256 sum_3 = _mm256_setzero_ps();

    for(size_t i(0); i < N; i += per_iteration, a += per_iteration, b += per_iteration){
      {
        const __m256 a_0 = _mm256_load_ps(a + 0*8); const __m256 b_0 = _mm256_load_ps(b + 0*8);
        sum_0 = _mm256_fmadd_ps(a_0, b_0, sum_0);
      }

      {
        const __m256 a_1 = _mm256_load_ps(a + 1*8); const __m256 b_1 = _mm256_load_ps(b + 1*8);
        sum_1 = _mm256_fmadd_ps(a_1, b_1, sum_1);
      }

      {
        const __m256 a_2 = _mm256_load_ps(a + 2*8); const __m256 b_2 = _mm256_load_ps(b + 2*8);
        sum_2 = _mm256_fmadd_ps(a_2, b_2, sum_2);
      }

      {
        const __m256 a_3 = _mm256_load_ps(a + 3*8); const __m256 b_3 = _mm256_load_ps(b + 3*8);
        sum_3 = _mm256_fmadd_ps(a_3, b_3, sum_3);
      }
    }

    const __m256 reduced_8  = _mm256_add_ps(_mm256_add_ps(sum_0, sum_1), _mm256_add_ps(sum_2, sum_3));
    // avoids extra move instruction by casting sum, adds lower 4 float elements to upper 4 float elements 
    const __m128 reduced_4 = _mm_add_ps(_mm256_castps256_ps128(reduced_8), _mm256_extractf128_ps(reduced_8, 1));
    // adds lower 2 float elements to the upper 2
    const __m128 reduced_2 = _mm_add_ps(reduced_4, _mm_movehl_ps(reduced_4, reduced_4));
    // adds 0th float element to 1st float element
    const __m128 reduced_1 = _mm_add_ss(reduced_2, _mm_shuffle_ps(reduced_2, reduced_2, 0x1));
    const T sum = _mm_cvtss_f32(reduced_1);
    return sum;

  }else{
    return dot_product_<T, N>(a, b);
  }
#else
  return dot_product_<T, N>(a, b);
#endif

}

}