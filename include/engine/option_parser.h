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

#include <engine/processor/types.h>
#include <util/tuple.h>

#include <functional>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace engine {

struct string_option {
  using type = std::string;
  static constexpr std::string_view empty = "<empty>";

  std::string name_;
  std::optional<std::string> default_{};

  template <typename F>
  [[nodiscard]] auto processor_for(F&& target) const noexcept {
    using namespace processor::def;
    return sequential(consume("setoption"), consume("name"), consume(name_), consume("value"), emit<type>, invoke(target));
  }

  explicit string_option(const std::string_view& name) noexcept : name_{name} {}
  string_option(const std::string_view& name, const std::string_view& def) noexcept : name_{name}, default_{def} {}
};

struct spin_range {
  int min, max;

  [[nodiscard]] constexpr int clamp(const int& x) const noexcept { return std::clamp(x, min, max); }
  constexpr spin_range(const int& a, const int& b) noexcept : min{a}, max{b} {}
};

struct spin_option {
  using type = int;

  std::string name_;
  std::optional<int> default_ = {};
  std::optional<spin_range> range_ = {};

  template <typename F>
  [[nodiscard]] auto processor_for(F&& target) const noexcept {
    using namespace processor::def;
    return sequential(consume("setoption"), consume("name"), consume(name_), consume("value"), emit<type>, invoke(target));
  }

  explicit spin_option(const std::string_view& name) noexcept : name_{name} {}
  spin_option(const std::string& name, const spin_range& range) noexcept : name_{name}, range_{range} {}
  spin_option(const std::string& name, const int& def) noexcept : name_{name}, default_{def} {}
  spin_option(const std::string& name, const int& def, const spin_range& range) noexcept : name_{name}, default_{def}, range_{range} {}
};

struct check_option {
  using type = bool;

  std::string name_;
  std::optional<bool> default_{std::nullopt};

  template <typename F>
  [[nodiscard]] auto processor_for(F&& target) const noexcept {
    // clang-format off

    using namespace processor::def;    
    return sequential(consume("setoption"), consume("name"), consume(name_), consume("value"), parallel(
      sequential(consume("true"), invoke([=] { target(true); })),
      sequential(consume("false"), invoke([=] { target(false); }))
    ));

    // clang-format on
  }

  explicit check_option(const std::string_view& name) noexcept : name_{name} {}
  check_option(const std::string_view& name, const bool& def) noexcept : name_{name}, default_{def} {}
};

std::ostream& operator<<(std::ostream& ostr, const string_option& opt) noexcept;
std::ostream& operator<<(std::ostream& ostr, const spin_option& opt) noexcept;
std::ostream& operator<<(std::ostream& ostr, const check_option& opt) noexcept;

template <typename T>
inline constexpr bool is_option_v = std::is_same_v<T, spin_option> || std::is_same_v<T, string_option> || std::is_same_v<T, check_option>;

template <typename T>
struct option_callback {
  static_assert(is_option_v<T>, "T must be of option type");

  T option_;
  std::function<void(typename T::type)> callback_;

  auto processor() const noexcept { return option_.processor_for(callback_); }

  template <typename F>
  option_callback(const T& option, F&& f) noexcept : option_{option}, callback_{f} {}
};

template <typename... Ts>
struct uci_options {
  std::tuple<option_callback<Ts>...> options_;

  auto processor() const noexcept {
    auto convert = [](const auto& callback) { return callback.processor(); };

    auto convert_many = [&convert](const auto&... callbacks) {
      using namespace processor::def;
      return parallel(convert(callbacks)...);
    };

    return std::apply(convert_many, options_);
  }

  explicit uci_options(const option_callback<Ts>&... options) noexcept : options_{options...} {}
};

template <typename... Ts>
inline std::ostream& operator<<(std::ostream& ostr, uci_options<Ts...> options) noexcept {
  util::tuple::for_each(options.options_, [&ostr](const auto& opt) { ostr << opt.option_ << std::endl; });
  return ostr;
}

}  // namespace engine
