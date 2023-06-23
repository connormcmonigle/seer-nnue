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
#include <cstring>
#include <iostream>
#include <type_traits>
#include <utility>

namespace nnue {

template <typename T>
struct dot_type_impl {};

template <>
struct dot_type_impl<float> {
  using type = float;
};

template <>
struct dot_type_impl<double> {
  using type = double;
};

template <>
struct dot_type_impl<std::int8_t> {
  using type = std::int16_t;
};

template <>
struct dot_type_impl<std::int16_t> {
  using type = std::int32_t;
};

template <>
struct dot_type_impl<std::int32_t> {
  using type = std::int64_t;
};

template <typename T>
using dot_type = typename dot_type_impl<T>::type;

template <typename T, size_t dim>
struct aligned_slice {
  T* data;

  template <size_t out_dim, size_t offset = 0>
  aligned_slice<T, out_dim> slice() {
    static_assert(offset + out_dim <= dim);
    return aligned_slice<T, out_dim>{data + offset};
  }

  aligned_slice<T, dim>& copy_from(const T* other) {
    std::memcpy(data, other, sizeof(T) * dim);
    return *this;
  }

  aligned_slice<T, dim>& copy_from(const aligned_slice<T, dim>& other) {
    std::memcpy(data, other.data, sizeof(T) * dim);
    return *this;
  }

  aligned_slice<T, dim>& store_summand(const aligned_slice<T, dim>& a, const aligned_slice<T, dim>& b) {
    simd::add_add<dim>(a.data, b.data, data);
    return *this;
  }

  aligned_slice(T* data) : data{data} {}
};

template <typename T, size_t dim>
std::ostream& operator<<(std::ostream& ostr, const aligned_slice<T, dim>& vec) {
  static_assert(dim != 0, "can't stream empty slice.");
  ostr << "aligned_slice<T, " << dim << ">([";
  for (size_t i = 0; i < (dim - 1); ++i) { ostr << vec.data[i] << ", "; }
  ostr << vec.data[dim - 1] << "])";
  return ostr;
}

template <typename T, size_t scratchpad_size>
struct stack_scratchpad {
  alignas(simd::alignment) T data[scratchpad_size];

  template <size_t dim>
  aligned_slice<T, dim> get_nth_slice(const size_t& n) {
    static_assert(scratchpad_size % dim == 0);
    return aligned_slice<T, dim>(data + n * dim);
  }
};

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
    simd::add<dim>(data, other);
    return *this;
  }

  inline stack_vector<T, dim>& sub_(const T* other) {
    simd::sub<dim>(data, other);
    return *this;
  }

  inline stack_vector<T, dim>& set_(const T* other) {
#pragma omp simd
    for (size_t i = 0; i < dim; ++i) { data[i] = other[i]; }
    return *this;
  }

  inline aligned_slice<T, dim> as_slice() { return aligned_slice<T, dim>(data); }

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
struct stack_relu_affine {
  static constexpr size_t W_numel = dim0 * dim1;
  static constexpr size_t b_numel = dim1;

  alignas(simd::alignment) T W[W_numel];
  alignas(simd::alignment) dot_type<T> b[b_numel];

  constexpr size_t num_parameters() const { return W_numel + b_numel; }

  inline stack_vector<dot_type<T>, dim1> forward(const stack_vector<T, dim0>& x) const {
    auto result = stack_vector<dot_type<T>, dim1>::from(b);
    simd::relu_matrix_vector_product<dim0, dim1>(W, x.data, result.data);
    return result;
  }

  inline stack_vector<dot_type<T>, dim1> forward(const aligned_slice<T, dim0>& x) const {
    auto result = stack_vector<dot_type<T>, dim1>::from(b);
    simd::relu_matrix_vector_product<dim0, dim1>(W, x.data, result.data);
    return result;
  }

  template <typename streamer_type>
  stack_relu_affine<T, dim0, dim1>& load_(streamer_type& ws) {
    ws.template stream<T>(W, W_numel).template stream<dot_type<T>>(b, b_numel);
    return *this;
  }

  stack_relu_affine<T, dim0, dim1> half_input_flipped() const {
    static_assert(dim0 % 2 == 0);
    constexpr size_t half_dim0 = dim0 / 2;

    stack_relu_affine<T, dim0, dim1> result = *this;
    for (size_t i(0); i < W_numel; i += dim0) {
      for (size_t j(0); j < half_dim0; ++j) { std::iter_swap(result.W + i + j, result.W + half_dim0 + i + j); }
    }

    return result;
  }

  template <typename U>
  stack_relu_affine<U, dim0, dim1> quantized(const T& weight_scale, const T& bias_scale) const {
    static_assert(std::is_floating_point_v<T> && std::is_integral_v<U>);
    stack_relu_affine<U, dim0, dim1> result{};
#pragma omp simd
    for (size_t i = 0; i < W_numel; ++i) { result.W[i] = static_cast<U>(std::round(weight_scale * W[i])); }
    for (size_t i = 0; i < b_numel; ++i) { result.b[i] = static_cast<dot_type<U>>(std::round(bias_scale * b[i])); }
    return result;
  }
};

template <typename T, size_t dim0, size_t dim1>
struct big_affine {
  static constexpr size_t W_numel = dim0 * dim1;
  static constexpr size_t b_numel = dim1;

  T* W{nullptr};
  alignas(simd::alignment) T b[b_numel];

  constexpr size_t num_parameters() const { return W_numel + b_numel; }

  void insert_idx(const size_t idx, aligned_slice<T, b_numel> x) const {
    const T* mem_region = W + idx * dim1;
    simd::add<b_numel>(x.data, mem_region);
  }

  void erase_idx(const size_t idx, aligned_slice<T, b_numel> x) const {
    const T* mem_region = W + idx * dim1;
    simd::sub<b_numel>(x.data, mem_region);
  }

  void insert_erase_idx(
      const size_t insert_idx,
      const size_t erase_idx,
      const aligned_slice<T, b_numel>& src,
      aligned_slice<T, b_numel> delta,
      aligned_slice<T, b_numel> dst) const {
    const T* insert_mem_region = W + insert_idx * dim1;
    const T* erase_mem_region = W + erase_idx * dim1;
    simd::add_add_sub<b_numel>(src.data, insert_mem_region, erase_mem_region, delta.data, dst.data);
  }

  void insert_erase_erase_idx(
      const size_t insert_idx,
      const size_t erase_idx_0,
      const size_t erase_idx_1,
      const aligned_slice<T, b_numel>& src,
      aligned_slice<T, b_numel> delta,
      aligned_slice<T, b_numel> dst) const {
    const T* insert_mem_region = W + insert_idx * dim1;
    const T* erase_mem_region_0 = W + erase_idx_0 * dim1;
    const T* erase_mem_region_1 = W + erase_idx_1 * dim1;
    simd::add_add_sub_sub<b_numel>(src.data, insert_mem_region, erase_mem_region_0, erase_mem_region_1, delta.data, dst.data);
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
    W = static_cast<T*>(simd::aligned_alloc(simd::alignment, sizeof(T) * W_numel));
#pragma omp simd
    for (size_t i = 0; i < W_numel; ++i) { W[i] = other.W[i]; }
    for (size_t i = 0; i < b_numel; ++i) { b[i] = other.b[i]; }
  }

  big_affine(big_affine<T, dim0, dim1>&& other) {
    std::swap(W, other.W);
    std::swap(b, other.b);
  }

  big_affine() { W = static_cast<T*>(simd::aligned_alloc(simd::alignment, sizeof(T) * W_numel)); }
  ~big_affine() {
    if (W != nullptr) { simd::aligned_free(W); }
  }
};

}  // namespace nnue
