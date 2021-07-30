#pragma once
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