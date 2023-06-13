/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See
  the GNU General Public License for more details. You should have received a copy of the GNU General Public License along with this program.  If not,
  see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>

namespace nnue {

template <typename T, size_t dim>
struct feature_transformer_sparsity_stats {
  struct element_stats {
    size_t idx{};
    size_t gt_zero{};
  };

  static inline std::array<element_stats, dim> stats = [] {
    std::array<element_stats, dim> result{};
    for (size_t i(0); i < dim; ++i) { result[i].idx = i; }
    return result;
  }();

  static void update(const size_t& idx, const T& value) {
    if (value > T{}) { ++stats[idx].gt_zero; }
  }

  static std::array<size_t, dim> permuted_indices() {
    std::array<element_stats, dim> sorted = stats;
    std::sort(sorted.begin(), sorted.end(), [](const element_stats& a, const element_stats& b) { return a.gt_zero < b.gt_zero; });

    std::array<size_t, dim> indices{};
    std::transform(sorted.begin(), sorted.end(), indices.begin(), [](const element_stats& x) { return x.idx; });
    return indices;
  }

  void operator()(const size_t& idx, const T& value) const { feature_transformer_sparsity_stats::update(idx, value); }
};

}  // namespace nnue