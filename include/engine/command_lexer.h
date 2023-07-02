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

#include <util/string.h>

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace engine {

namespace lexer {
constexpr const char* delimiter = " ";
using token_type = std::string;
using token_sequence_type = std::vector<token_type>;
}  // namespace lexer

struct lexed_command_view {
  static constexpr std::string_view separator = " ";

  lexer::token_sequence_type::const_iterator begin_;
  lexer::token_sequence_type::const_iterator end_;

  [[nodiscard]] lexer::token_sequence_type tokens() const { return lexer::token_sequence_type(begin_, end_); }

  template <typename F>
  void consume(const lexer::token_type& token, F&& receiver) const noexcept {
    if (begin_ != end_ && *begin_ == token) {
      const lexed_command_view next_view{std::next(begin_), end_};
      receiver(next_view);
    }
  }

  template <typename T, typename F>
  void emit(F&& receiver) const noexcept {
    if (begin_ != end_) {
      std::istringstream stream(*begin_);

      T value{};
      stream >> value;

      const lexed_command_view next_view{std::next(begin_), end_};
      receiver(value, next_view);
    }
  }

  template <std::ptrdiff_t N, typename F>
  void emit_n(F&& receiver) const noexcept {
    if (std::distance(begin_, end_) >= N) {
      const auto it = begin_ + N;
      const lexed_command_view next_view{it, end_};
      receiver(util::string::join(begin_, it, std::string(separator)), next_view);
    }
  }

  template <typename F>
  void emit_all(F&& receiver) const noexcept {
    const lexed_command_view next_view{end_, end_};
    receiver(util::string::join(begin_, end_, std::string(separator)), next_view);
  }

  template <typename F>
  void extract_condition(const lexer::token_type& token, F&& receiver) const noexcept {
    const bool exists = std::find(begin_, end_, token) != end_;
    receiver(exists, *this);
  }

  template <typename T, typename F>
  void extract_key(const lexer::token_type& token, F&& receiver) const noexcept {
    const auto key_it = std::find(begin_, end_, token);
    if (key_it != end_ && std::next(key_it) != end_) {
      const auto value_it = std::next(key_it);
      std::istringstream stream(*value_it);

      T value{};
      stream >> value;

      receiver(value, *this);
    }
  }
};

struct lexed_command {
  lexer::token_sequence_type tokens_{};

  [[nodiscard]] lexed_command_view view() const noexcept { return lexed_command_view{tokens_.cbegin(), tokens_.cend()}; }

  explicit lexed_command(const std::string& input) noexcept {
    std::istringstream input_stream(input);
    std::copy(std::istream_iterator<lexer::token_type>(input_stream), std::istream_iterator<lexer::token_type>(), std::back_inserter(tokens_));
  }
};

struct command_lexer {
  [[nodiscard]] lexed_command lex(const std::string& input) const noexcept { return lexed_command(input); }
};

}  // namespace engine