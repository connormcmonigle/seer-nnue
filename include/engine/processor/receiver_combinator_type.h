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

namespace engine::processor {

template <typename A, typename B>
struct receiver_combinator_type {
  A processor_;
  B receiver_;

  template <typename... As>
  void process(const lexed_command_view& view, const std::tuple<As...>& args) const noexcept {
    processor_.process(view, args, receiver_);
  }

  constexpr receiver_combinator_type(const A& processor, const B& receiver) noexcept : processor_{processor}, receiver_{receiver} {}
};

}  // namespace engine::processor
