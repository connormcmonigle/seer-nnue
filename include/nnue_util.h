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
#include <simd.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <random>
#include <utility>

namespace nnue {

template <typename T>
constexpr T relu(const T& x) {
  return std::max(x, T{0});
}

template <typename T, size_t dim>
struct stack_vector {
  alignas(simd::alignment) T data[dim];

  template <typename F>
  constexpr stack_vector<T, dim> apply(F&& f) const {
    return stack_vector<T, dim>{*this}.apply_(std::forward<F>(f));
  }

  template <typename F>
  inline stack_vector<T, dim>& apply_(F&& f) {
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { data[i] = f(data[i]); }
    return *this;
  }

  inline stack_vector<T, dim>& softmax_() {
    static_assert(dim != 0, "can't softmax empty vector.");
    T maximum_value = data[0];
    for (size_t i = 0; i < dim; ++i) {
      if (data[i] > maximum_value) { maximum_value = data[i]; }
    }
    apply_([maximum_value](const T& x) { return std::exp(x - maximum_value); });
    const T z = sum();
    apply_([z](const T& x) { return x / z; });
    return *this;
  }

  inline stack_vector<T, dim>& add_(const T* other) {
    for (size_t i = 0; i < dim; ++i) { data[i] += other[i]; }
    return *this;
  }

  inline stack_vector<T, dim>& sub_(const T* other) {
    for (size_t i = 0; i < dim; ++i) { data[i] -= other[i]; }
    return *this;
  }

  inline stack_vector<T, dim>& set_(const T* other) {
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { data[i] = other[i]; }
    return *this;
  }

  template <typename U>
  inline stack_vector<U, dim> mul_convert(const U& mul) const {
    stack_vector<U, dim> result{};
    simd::mul_convert<dim>(data, result.data, mul);
    return result;
  }

  inline util::dot_type<T> dot(const T* other) const { return simd::dot_product<dim>(data, other); }

  inline T item() const {
    static_assert(dim == 1, "called item() on vector with dim != 1");
    return data[0];
  }

  inline T sum() const {
    T result{};
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { result += data[i]; }
    return result;
  }

  static inline stack_vector<T, dim> zeros() {
    stack_vector<T, dim> result{};
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { result.data[i] = T(0); }
    return result;
  }

  static inline stack_vector<T, dim> ones() {
    stack_vector<T, dim> result{};
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { result.data[i] = T(1); }
    return result;
  }

  static inline stack_vector<T, dim> from(const T* data) {
    stack_vector<T, dim> result{};
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { result.data[i] = data[i]; }
    return result;
  }
};

template <typename T, size_t dim>
std::ostream& operator<<(std::ostream& ostr, const stack_vector<T, dim>& vec) {
  static_assert(dim != 0, "can't stream empty vector.");
  ostr << "stack_vector<T, " << dim << ">([";
  for (size_t i = 0; i < (dim - 1); ++i) { ostr << vec.data[i] << ", "; }
  ostr << vec.data[dim - 1] << "])";
  return ostr;
}

template <typename T, size_t dim0, size_t dim1>
inline stack_vector<T, dim0 + dim1> splice(const stack_vector<T, dim0>& a, const stack_vector<T, dim1>& b) {
  auto c = stack_vector<T, dim0 + dim1>::zeros();
#pragma omp simd
  for (size_t i = 0; i < dim0; ++i) { c.data[i] = a.data[i]; }
  for (size_t i = 0; i < dim1; ++i) { c.data[dim0 + i] = b.data[i]; }
  return c;
}

template <typename T, size_t dim0, size_t dim1>
struct stack_affine {
  using weight_type = T;
  using bias_type = util::dot_type<T>;

  static constexpr size_t W_numel = dim0 * dim1;
  static constexpr size_t b_numel = dim1;

  alignas(simd::alignment) weight_type W[W_numel];
  alignas(simd::alignment) bias_type b[b_numel];

  constexpr size_t num_parameters() const { return W_numel + b_numel; }

  inline stack_vector<bias_type, dim1> forward(const stack_vector<weight_type, dim0>& x) const {
    auto result = stack_vector<bias_type, dim1>::from(b);
    for (size_t i = 0; i < dim1; ++i) { result.data[i] += x.dot(W + i * dim0); }
    return result;
  }

  template <typename streamer_type>
  stack_affine<T, dim0, dim1>& load_(streamer_type& ws) {
    ws.stream(W, W_numel).stream(b, b_numel);
    return *this;
  }
};

template <typename T, size_t dim0, size_t dim1>
struct big_affine {
  using weight_type = T;
  using bias_type = T;

  static constexpr size_t W_numel = dim0 * dim1;
  static constexpr size_t b_numel = dim1;

  T* W{nullptr};
  alignas(simd::alignment) T b[b_numel];

  constexpr size_t num_parameters() const { return W_numel + b_numel; }

  void insert_idx(const size_t idx, stack_vector<T, b_numel>& x) const {
    const T* mem_region = W + idx * dim1;
    x.add_(mem_region);
  }

  void erase_idx(const size_t idx, stack_vector<T, b_numel>& x) const {
    const T* mem_region = W + idx * dim1;
    x.sub_(mem_region);
  }

  template <typename streamer_type>
  big_affine<T, dim0, dim1>& load_(streamer_type& ws) {
    ws.stream(W, W_numel).stream(b, b_numel);
    return *this;
  }

  big_affine<T, dim0, dim1>& operator=(const big_affine<T, dim0, dim1>& other) {
#pragma omp simd
    for (size_t i = 0; i < W_numel; ++i) { W[i] = other.W[i]; }
    for (size_t i = 0; i < b_numel; ++i) { b[i] = other.b[i]; }
    return *this;
  }

  big_affine<T, dim0, dim1>& operator=(big_affine<T, dim0, dim1>&& other) {
    std::swap(W, other.W);
    std::swap(b, other.b);
    return *this;
  }

  big_affine(const big_affine<T, dim0, dim1>& other) {
    W = new T[W_numel];
#pragma omp simd
    for (size_t i = 0; i < W_numel; ++i) { W[i] = other.W[i]; }
    for (size_t i = 0; i < b_numel; ++i) { b[i] = other.b[i]; }
  }

  big_affine(big_affine<T, dim0, dim1>&& other) {
    std::swap(W, other.W);
    std::swap(b, other.b);
  }

  big_affine() { W = new T[W_numel]; }
  ~big_affine() {
    if (W != nullptr) { delete[] W; }
  }
};

template <typename A, typename B>
void quantized_affine(const A& a, B& b, const typename A::weight_type& mul) {
  static_assert(A::W_numel == B::W_numel);
  static_assert(A::b_numel == B::b_numel);
  static_assert(std::is_floating_point_v<typename A::weight_type>);
  static_assert(std::is_floating_point_v<typename A::bias_type>);
  static_assert(std::is_integral_v<typename B::weight_type>);
  static_assert(std::is_integral_v<typename B::bias_type>);

  using weight_type = typename B::weight_type;
  using bias_type = typename B::bias_type;

  for (size_t i = 0; i < A::W_numel; ++i) { b.W[i] = static_cast<weight_type>(mul * a.W[i]); }
  for (size_t i = 0; i < A::b_numel; ++i) { b.b[i] = static_cast<bias_type>(mul * a.b[i]); }
}

}  // namespace nnue
