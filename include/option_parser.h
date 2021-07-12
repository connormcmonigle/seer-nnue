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

#include <apply.h>

#include <functional>
#include <iostream>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>

namespace engine {

struct string_option {
  using type = std::string;
  std::string name_;
  std::optional<std::string> default_ = {};

  std::optional<std::string> maybe_read(const std::string& cmd) const {
    std::regex regex("setoption name " + name_ + " value (.*)");
    if (auto matches = std::smatch{}; std::regex_search(cmd, matches, regex)) {
      std::string match = matches.str(1);
      if (match == "") {
        return default_;
      } else {
        return match;
      }
    }
    return std::nullopt;
  }

  string_option(const std::string_view& name) : name_{name} {}
  string_option(const std::string_view& name, const std::string& def) : name_{name}, default_{def} {}
};

struct spin_range {
  int min, max;

  int clamp(const int& x) const { return std::clamp(x, min, max); }

  spin_range(const int& a, const int& b) : min{a}, max{b} {}
};

struct spin_option {
  using type = int;
  std::string name_;
  std::optional<int> default_ = {};
  std::optional<spin_range> range_ = {};

  std::optional<int> maybe_read(const std::string& cmd) const {
    std::regex regex("setoption name " + name_ + " value (-?[0-9]+)");
    if (auto matches = std::smatch{}; std::regex_search(cmd, matches, regex)) {
      const int raw = std::stoi(matches.str(1));
      if (range_.has_value()) {
        return range_.value().clamp(raw);
      } else {
        return raw;
      }
    }
    return std::nullopt;
  }

  spin_option(const std::string_view& name) : name_{name} {}
  spin_option(const std::string& name, const spin_range& range) : name_{name}, range_{range} {}
  spin_option(const std::string& name, const int& def) : name_{name}, default_{def} {}
  spin_option(const std::string& name, const int& def, const spin_range& range) : name_{name}, default_{def}, range_{range} {}
};

struct button_option {
  using type = bool;
  std::string name_;

  std::optional<bool> maybe_read(const std::string& cmd) const {
    if (cmd == (std::string("setoption name ") + name_)) {
      return true;
    } else {
      return std::nullopt;
    }
  }

  button_option(const std::string_view& name) : name_{name} {}
};

struct check_option {
  using type = bool;
  std::string name_;
  std::optional<bool> default_{std::nullopt};

  std::optional<bool> maybe_read(const std::string& cmd) const {
    std::regex regex("setoption name " + name_ + " value (true|false)");
    if (auto matches = std::smatch{}; std::regex_search(cmd, matches, regex)) { return "true" == matches.str(1); }
    return std::nullopt;
  }

  check_option(const std::string_view& name) : name_{name} {}

  check_option(const std::string_view& name, const bool& def) : name_{name}, default_{def} {}
};

std::ostream& operator<<(std::ostream& ostr, const string_option& opt) {
  ostr << "option name " << opt.name_ << " type string";
  if (opt.default_.has_value()) { ostr << " default " << opt.default_.value(); }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const spin_option& opt) {
  ostr << "option name " << opt.name_ << " type spin";
  if (opt.default_.has_value()) { ostr << " default " << opt.default_.value(); }
  if (opt.range_.has_value()) { ostr << " min " << opt.range_.value().min << " max " << opt.range_.value().max; }
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const button_option& opt) {
  ostr << "option name " << opt.name_ << " type button";
  return ostr;
}

std::ostream& operator<<(std::ostream& ostr, const check_option& opt) {
  ostr << "option name " << opt.name_ << " type check";
  if (opt.default_.has_value()) { ostr << std::boolalpha << " default " << opt.default_.value(); }
  return ostr;
}

template <typename T>
inline constexpr bool is_option_v =
    std::is_same_v<T, spin_option> || std::is_same_v<T, string_option> || std::is_same_v<T, button_option> || std::is_same_v<T, check_option>;

template <typename T>
struct option_callback {
  static_assert(is_option_v<T>, "T must be of option type");

  T option_;
  std::function<void(typename T::type)> callback_;

  void maybe_call(const std::string& cmd) {
    std::optional<typename T::type> read = option_.maybe_read(cmd);
    if (read.has_value()) { callback_(read.value()); }
  }

  template <typename F>
  option_callback(const T& option, F&& f) : option_{option}, callback_{f} {}
};

template <typename... Ts>
struct uci_options {
  std::tuple<option_callback<Ts>...> options_;

  void update(const std::string& cmd) {
    util::apply(options_, [cmd](auto& opt) { opt.maybe_call(cmd); });
  }

  uci_options(const option_callback<Ts>&... options) : options_{options...} {}
};

template <typename... Ts>
std::ostream& operator<<(std::ostream& ostr, uci_options<Ts...> options) {
  util::apply(options.options_, [&ostr](const auto& opt) { ostr << opt.option_ << std::endl; });
  return ostr;
}

}  // namespace engine
