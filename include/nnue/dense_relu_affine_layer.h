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
#include <nnue/aligned_vector.h>
#include <nnue/dot_type.h>
#include <nnue/simd.h>

#include <cstddef>

namespace nnue {

template <typename T, std::size_t dim0, std::size_t dim1>
struct dense_relu_affine_layer {
  static constexpr std::size_t W_numel = dim0 * dim1;
  static constexpr std::size_t b_numel = dim1;

  alignas(simd::alignment) T W[W_numel];
  alignas(simd::alignment) dot_type<T> b[b_numel];

  [[nodiscard]] constexpr std::size_t num_parameters() const noexcept { return W_numel + b_numel; }

  [[nodiscard]] inline aligned_vector<dot_type<T>, dim1> forward(const aligned_vector<T, dim0>& x) const noexcept {
    auto result = aligned_vector<dot_type<T>, dim1>::from(b);
    simd::relu_matrix_vector_product<dim0, dim1>(W, x.data, result.data);
    return result;
  }

  [[nodiscard]] inline aligned_vector<dot_type<T>, dim1> forward(const aligned_slice<T, dim0>& x) const noexcept {
    auto result = aligned_vector<dot_type<T>, dim1>::from(b);
    simd::relu_matrix_vector_product<dim0, dim1>(W, x.data, result.data);
    return result;
  }

  template <typename streamer_type>
  [[maybe_unused]] dense_relu_affine_layer<T, dim0, dim1>& load_(streamer_type& streamer) noexcept {
    streamer.template stream<T>(W, W_numel).template stream<dot_type<T>>(b, b_numel);
    return *this;
  }

  template <typename exporter_type>
  [[maybe_unused]] const dense_relu_affine_layer<T, dim0, dim1>& write_(exporter_type& exporter) const noexcept {
    exporter.template write<T>(W, W_numel).template write<dot_type<T>>(b, b_numel);
    return *this;
  }

  [[nodiscard]] dense_relu_affine_layer<T, dim0, dim1> half_input_flipped() const noexcept {
    static_assert(dim0 % 2 == 0);
    constexpr std::size_t half_dim0 = dim0 / 2;

    dense_relu_affine_layer<T, dim0, dim1> result = *this;
    for (std::size_t i(0); i < W_numel; i += dim0) {
      for (std::size_t j(0); j < half_dim0; ++j) { std::iter_swap(result.W + i + j, result.W + half_dim0 + i + j); }
    }

    return result;
  }

  template <typename U>
  [[nodiscard]] dense_relu_affine_layer<U, dim0, dim1> quantized(const T& weight_scale, const T& bias_scale) const noexcept {
    static_assert(std::is_floating_point_v<T> && std::is_integral_v<U>);
    dense_relu_affine_layer<U, dim0, dim1> result{};
#pragma omp simd
    for (std::size_t i = 0; i < W_numel; ++i) { result.W[i] = static_cast<U>(std::round(weight_scale * W[i])); }
    for (std::size_t i = 0; i < b_numel; ++i) { result.b[i] = static_cast<dot_type<U>>(std::round(bias_scale * b[i])); }
    return result;
  }
};

}  // namespace nnue
