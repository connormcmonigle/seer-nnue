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

#include <simd.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <type_traits>
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

  inline T dot(const T* other) const { return simd::dot_product<dim>(data, other); }

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

  template <typename U>
  inline stack_vector<U, dim> dequantized(const U& scale) const {
    static_assert(std::is_integral_v<T> && std::is_floating_point_v<U>);
    stack_vector<U, dim> result;
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { result.data[i] = scale * static_cast<U>(data[i]); }
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
  static constexpr size_t W_numel = dim0 * dim1;
  static constexpr size_t b_numel = dim1;

  alignas(simd::alignment) T W[W_numel];
  alignas(simd::alignment) T b[b_numel];

  constexpr size_t num_parameters() const { return W_numel + b_numel; }

  inline stack_vector<T, dim1> forward(const stack_vector<T, dim0>& x) const {
    auto result = stack_vector<T, dim1>::from(b);
    simd::matrix_vector_product<dim0, dim1>(W, x.data, result.data);
    return result;
  }

  template <typename streamer_type>
  stack_affine<T, dim0, dim1>& load_(streamer_type& ws) {
    ws.template stream<T>(W, W_numel).template stream<T>(b, b_numel);
    return *this;
  }
};

template <typename T, size_t dim0, size_t dim1>
struct big_affine {
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
    ws.template stream<T>(W, W_numel).template stream<T>(b, b_numel);
    return *this;
  }

  template <typename U>
  big_affine<U, dim0, dim1> quantized(const T& scale) const {
    static_assert(std::is_floating_point_v<T> && std::is_integral_v<U>);
    big_affine<U, dim0, dim1> result{};
#pragma omp simd
    for (size_t i = 0; i < W_numel; ++i) { result.W[i] = static_cast<U>(std::round(scale * W[i])); }
    for (size_t i = 0; i < b_numel; ++i) { result.b[i] = static_cast<U>(std::round(scale * b[i])); }
    return result;
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

}  // namespace nnue
