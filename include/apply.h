#pragma once
#include <tuple>

namespace util {

template <typename... Ts, typename F, size_t... I>
void apply_impl(std::tuple<Ts...>& data, F&& f, std::index_sequence<I...>) {
  auto map = [&f](auto&& x) {
    f(x);
    return 0;
  };
  [[maybe_unused]] const auto ignored = {map(std::get<I>(data))...};
}

template <typename... Ts, typename F>
void apply(std::tuple<Ts...>& data, F&& f) {
  apply_impl(data, std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{});
}

template <typename... Ts, typename F, size_t... I>
void apply_impl(const std::tuple<Ts...>& data, F&& f, std::index_sequence<I...>) {
  auto map = [&f](auto&& x) {
    f(x);
    return 0;
  };
  [[maybe_unused]] const auto ignored = {map(std::get<I>(data))...};
}

template <typename... Ts, typename F>
void apply(const std::tuple<Ts...>& data, F&& f) {
  apply_impl(data, std::forward<F>(f), std::make_index_sequence<sizeof...(Ts)>{});
}


}  // namespace util