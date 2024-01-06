
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

#include <engine/command_lexer.h>
#include <engine/processor/null_type.h>
#include <util/tuple.h>

#include <tuple>

namespace engine::processor {

template <typename T>
struct key_type {
  lexer::token_type token_{};

  template <typename... As, typename F = null_type>
  void process(const lexed_command_view& view, const std::tuple<As...>& args, const F& receiver = def::null) const noexcept {
    view.extract_key<T>(token_, [&](const T& value, const lexed_command_view& next_view) {
      const auto next_args = util::tuple::append(args, value);
      receiver.process(next_view, next_args);
    });
  }

  explicit key_type(const lexer::token_type& token) noexcept : token_{token} {}
};

namespace def {

template <typename T>
[[nodiscard]] inline key_type<T> key(const lexer::token_type& token) noexcept {
  return key_type<T>(token);
}

}  // namespace def

}  // namespace engine::processor
