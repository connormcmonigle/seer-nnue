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
#include <tuple>
#include <type_traits>

namespace util::tuple {

template <typename T, typename F, std::size_t... I>
constexpr void for_each_impl(T&& data, F&& f, std::index_sequence<I...>) noexcept {
  auto map = [&f](auto&& x) {
    f(x);
    return 0;
  };

  [[maybe_unused]] const auto ignored = {map(std::get<I>(data))...};
}

template <typename T, typename F>
constexpr void for_each(T&& data, F&& f) noexcept {
  for_each_impl(std::forward<T>(data), std::forward<F>(f), std::make_index_sequence<std::tuple_size_v<std::decay_t<T>>>{});
}

template <typename T, typename... Ts, std::size_t I, std::size_t... Is>
constexpr std::tuple<Ts...> tail_impl(const std::tuple<T, Ts...>& tuple, std::index_sequence<I, Is...>) noexcept {
  return std::tuple{std::get<Is>(tuple)...};
}

template <typename T, typename... Ts>
constexpr std::tuple<Ts...> tail(const std::tuple<T, Ts...>& tuple) noexcept {
  const auto sequence = std::make_index_sequence<std::tuple_size_v<std::tuple<T, Ts...>>>{};
  return tail_impl(tuple, sequence);
}

template <typename T, typename... Ts>
constexpr T head(const std::tuple<T, Ts...>& tuple) noexcept {
  return std::get<0>(tuple);
}

template <typename... Ts, typename... As>
constexpr std::tuple<Ts..., As...> append(const std::tuple<Ts...>& tuple, const As&... args) noexcept {
  return std::tuple_cat(tuple, std::tuple{args...});
}

}  // namespace util::tuple