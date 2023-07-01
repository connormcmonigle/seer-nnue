
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

#include <tuple>

namespace engine {
namespace processor {

struct null_type {
  template <typename... As>
  void process(const lexed_command_view&, const std::tuple<As...>&) const noexcept {}

  template <typename... As, typename F>
  void process(const lexed_command_view&, const std::tuple<As...>&, const F&) const noexcept {}
};

namespace def {

constexpr auto null = null_type{};

}

}  // namespace processor
}  // namespace engine