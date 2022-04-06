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

#include <cstddef>
#include <tuple>
#include <type_traits>

namespace util {

template <typename T, typename F, size_t... I>
constexpr void apply_impl(T&& data, F&& f, std::index_sequence<I...>) {
  auto map = [&f](auto&& x) {
    f(x);
    return 0;
  };
  [[maybe_unused]] const auto ignored = {map(std::get<I>(data))...};
}

template <typename T, typename F>
constexpr void apply(T&& data, F&& f) {
  apply_impl(std::forward<T>(data), std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<std::decay_t<T>>>{});
}

}  // namespace util