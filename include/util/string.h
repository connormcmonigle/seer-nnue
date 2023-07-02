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
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <type_traits>

namespace util::string {

template <typename T>
[[nodiscard]] std::string join(const T& first, const T& last, const std::string& separator) noexcept {
  std::ostringstream result{};
  std::copy(first, last, std::ostream_iterator<std::string>(result, separator.c_str()));
  return result.str();
}

}  // namespace util::string