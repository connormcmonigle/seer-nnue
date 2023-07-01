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
#include <engine/processor/parallel_combinator_type.h>
#include <engine/processor/receiver_combinator_type.h>

#include <tuple>

namespace engine {
namespace processor {

template <typename T, typename... Ts>
struct sequential_combinator_type {
  std::tuple<T, Ts...> processors_;

  template <typename... As, typename F = null_type>
  void process(const lexed_command_view& view, const std::tuple<As...>& args, const F& receiver = def::null) const noexcept {
    if constexpr (std::is_same_v<std::tuple<>, std::tuple<Ts...>>) {
      const auto head = util::tuple::head(processors_);
      head.process(view, args, receiver);
    } else {
      const auto head = util::tuple::head(processors_);
      const auto tail = sequential_combinator_type<Ts...>(util::tuple::tail(processors_));
      const auto next_receiver = receiver_combinator_type(tail, receiver);
      head.process(view, args, next_receiver);
    }
  }

  constexpr sequential_combinator_type(const std::tuple<T, Ts...>& processors) noexcept : processors_(processors) {}
};

namespace def {

template <typename... Ts>
[[nodiscard]] constexpr auto sequential(Ts&&... ts) noexcept {
  const auto processors = std::tuple{std::forward<Ts>(ts)...};
  return sequential_combinator_type(processors);
}

}  // namespace def

}  // namespace processor
}  // namespace engine