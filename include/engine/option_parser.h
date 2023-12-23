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
#include <sstream>
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

template <typename T>
struct value_range {
  T min, max;

  [[nodiscard]] constexpr T clamp(const T& x) const noexcept { return std::clamp(x, min, max); }
  constexpr value_range(const T& a, const T& b) noexcept : min{a}, max{b} {}
};

using spin_range = value_range<int>;

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

template <typename T>
struct tune_option {
  using type = T;

  std::string name_;
  type default_;
  value_range<type> range_;

  double c_end_{1.0};
  double r_end_{0.002};

  [[nodiscard]] constexpr double c_end() const noexcept { return c_end_; }
  [[nodiscard]] constexpr double r_end() const noexcept { return r_end_; }

  template <typename F>
  [[nodiscard]] auto processor_for(F&& callback) const noexcept {
    using namespace processor::def;
    return sequential(consume("setoption"), consume("name"), consume(name_), consume("value"), emit<type>, invoke(callback));
  }

  [[maybe_unused]] tune_option<T>& set_c_end(const double& value) noexcept {
    c_end_ = value;
    return *this;
  }

  [[maybe_unused]] tune_option<T>& set_r_end(const double& value) noexcept {
    r_end_ = value;
    return *this;
  }

  [[nodiscard]] std::string ob_spsa_config() const noexcept {
    std::stringstream config{};
    config << name_ << ", ";

    if (std::is_floating_point_v<T>) {
      config << "float";
    } else if (std::is_integral_v<T>) {
      config << "int";
    }

    config << ", ";
    config << default_ << ", ";
    config << range_.min << ", ";
    config << range_.max << ", ";
    config << c_end_ << ", ";
    config << r_end_;

    return config.str();
  }

  tune_option(const std::string& name, const T& def, const value_range<T>& range) noexcept : name_{name}, default_{def}, range_{range} {}
};

using tune_float_option = tune_option<double>;
using tune_int_option = tune_option<int>;

std::ostream& operator<<(std::ostream& ostr, const string_option& opt) noexcept;
std::ostream& operator<<(std::ostream& ostr, const spin_option& opt) noexcept;
std::ostream& operator<<(std::ostream& ostr, const check_option& opt) noexcept;
std::ostream& operator<<(std::ostream& ostr, const tune_float_option& opt) noexcept;
std::ostream& operator<<(std::ostream& ostr, const tune_int_option& opt) noexcept;

template <typename T>
inline constexpr bool is_option_v = std::is_same_v<T, spin_option> || std::is_same_v<T, string_option> || std::is_same_v<T, check_option> ||
                                    std::is_same_v<T, tune_int_option> || std::is_same_v<T, tune_float_option>;

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

  [[nodiscard]] std::string ob_spsa_config() const noexcept {
    std::stringstream config{};
    util::tuple::for_each(options_, [&config](const auto& opt) { config << opt.option_.ob_spsa_config() << std::endl; });
    return config.str();
  }

  explicit uci_options(const option_callback<Ts>&... options) noexcept : options_{options...} {}
};

template <typename... Ts>
inline std::ostream& operator<<(std::ostream& ostr, uci_options<Ts...> options) noexcept {
  util::tuple::for_each(options.options_, [&ostr](const auto& opt) { ostr << opt.option_ << std::endl; });
  return ostr;
}

}  // namespace engine
