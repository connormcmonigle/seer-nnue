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

#include <cstddef>
#include <cstdlib>
#include <new>

#if defined(__linux__)
#include <sys/mman.h>
#endif

#if defined(_WIN32)
#include <malloc.h>
#endif

namespace util {

constexpr std::size_t huge_page_size = std::size_t{1} << 21;

[[nodiscard]] inline void* huge_page_aligned_alloc(const std::size_t& size) noexcept {
  const std::size_t padded_size = ((size + huge_page_size - 1) / huge_page_size) * huge_page_size;
#if defined(_WIN32)
  return _mm_malloc(padded_size, huge_page_size);
#else
  void* ptr = std::aligned_alloc(huge_page_size, padded_size);
#if defined(__linux__) && defined(MADV_HUGEPAGE)
  if (ptr != nullptr) { madvise(ptr, padded_size, MADV_HUGEPAGE); }
#endif
  return ptr;
#endif
}

inline void huge_page_aligned_free(void* ptr) noexcept {
#if defined(_WIN32)
  _mm_free(ptr);
#else
  std::free(ptr);
#endif
}

template <typename T>
struct huge_page_aligned_deleter {
  void operator()(T* ptr) const noexcept { huge_page_aligned_free(ptr); }
};

}  // namespace util
