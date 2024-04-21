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

#include <nnue/aligned_slice.h>
#include <nnue/simd.h>

#include <cstddef>

namespace nnue {

template <typename T, std::size_t dim0, std::size_t dim1>
struct sparse_affine_layer {
  static constexpr std::size_t W_numel = dim0 * dim1;
  static constexpr std::size_t b_numel = dim1;

  T* W{nullptr};
  alignas(simd::alignment) T b[b_numel];

  [[nodiscard]] constexpr std::size_t num_parameters() const { return W_numel + b_numel; }

  inline void insert_idx(const std::size_t idx, aligned_slice<T, b_numel> x) const {
    const T* mem_region = W + idx * dim1;
    simd::add<b_numel>(x.data, mem_region);
  }

  inline void erase_idx(const std::size_t idx, aligned_slice<T, b_numel> x) const {
    const T* mem_region = W + idx * dim1;
    simd::sub<b_numel>(x.data, mem_region);
  }

  inline void insert_erase_idx(
      const std::size_t insert_idx,
      const std::size_t erase_idx,
      const aligned_slice<T, b_numel>& src,
      aligned_slice<T, b_numel> dst) const {
    const T* insert_mem_region = W + insert_idx * dim1;
    const T* erase_mem_region = W + erase_idx * dim1;
    simd::add_add_sub<b_numel>(src.data, insert_mem_region, erase_mem_region, dst.data);
  }

  inline void insert_erase_erase_idx(
      const std::size_t insert_idx,
      const std::size_t erase_idx_0,
      const std::size_t erase_idx_1,
      const aligned_slice<T, b_numel>& src,
      aligned_slice<T, b_numel> dst) const {
    const T* insert_mem_region = W + insert_idx * dim1;
    const T* erase_mem_region_0 = W + erase_idx_0 * dim1;
    const T* erase_mem_region_1 = W + erase_idx_1 * dim1;
    simd::add_add_sub_sub<b_numel>(src.data, insert_mem_region, erase_mem_region_0, erase_mem_region_1, dst.data);
  }

  template <typename streamer_type>
  sparse_affine_layer<T, dim0, dim1>& load_(streamer_type& streamer) noexcept {
    streamer.template stream<T>(W, W_numel).template stream<T>(b, b_numel);
    return *this;
  }

  template <typename exporter_type>
  const sparse_affine_layer<T, dim0, dim1>& write_(exporter_type& exporter) const noexcept {
    exporter.template write<T>(W, W_numel).template write<T>(b, b_numel);
    return *this;
  }

  template <typename U>
  sparse_affine_layer<U, dim0, dim1> quantized(const T& scale) const {
    static_assert(std::is_floating_point_v<T> && std::is_integral_v<U>);
    sparse_affine_layer<U, dim0, dim1> result{};
#pragma omp simd
    for (std::size_t i = 0; i < W_numel; ++i) { result.W[i] = static_cast<U>(std::round(scale * W[i])); }
    for (std::size_t i = 0; i < b_numel; ++i) { result.b[i] = static_cast<U>(std::round(scale * b[i])); }
    return result;
  }

  sparse_affine_layer<T, dim0, dim1>& operator=(const sparse_affine_layer<T, dim0, dim1>& other) {
#pragma omp simd
    for (std::size_t i = 0; i < W_numel; ++i) { W[i] = other.W[i]; }
    for (std::size_t i = 0; i < b_numel; ++i) { b[i] = other.b[i]; }
    return *this;
  }

  sparse_affine_layer<T, dim0, dim1>& operator=(sparse_affine_layer<T, dim0, dim1>&& other) noexcept {
    std::swap(W, other.W);
    std::swap(b, other.b);
    return *this;
  }

  sparse_affine_layer(const sparse_affine_layer<T, dim0, dim1>& other) {
    W = static_cast<T*>(simd::aligned_alloc(simd::alignment, sizeof(T) * W_numel));
#pragma omp simd
    for (std::size_t i = 0; i < W_numel; ++i) { W[i] = other.W[i]; }
    for (std::size_t i = 0; i < b_numel; ++i) { b[i] = other.b[i]; }
  }

  sparse_affine_layer(sparse_affine_layer<T, dim0, dim1>&& other) noexcept {
    std::swap(W, other.W);
    std::swap(b, other.b);
  }

  sparse_affine_layer() { W = static_cast<T*>(simd::aligned_alloc(simd::alignment, sizeof(T) * W_numel)); }
  ~sparse_affine_layer() {
    if (W != nullptr) { simd::aligned_free(W); }
  }
};

}  // namespace nnue
