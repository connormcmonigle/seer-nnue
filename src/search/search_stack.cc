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

#include <search/search_stack.h>

#include <algorithm>

namespace search {

std::size_t search_stack::count(const std::size_t& height, const zobrist::hash_type& hash) const noexcept {
  const std::size_t future_count =
      std::count_if(future_.cbegin(), future_.cbegin() + height, [&](const stack_entry& entry) { return hash == entry.hash_; });

  return future_count + past_.count(hash);
}

std::string search_stack::pv_string() const noexcept {
  auto bd = present_;
  std::string result{};
  
  for (const auto& pv_mv : future_.begin()->pv_) {
    if (!bd.generate_moves<>().has(pv_mv)) { break; }
    result += pv_mv.name(bd.turn()) + " ";
    bd = bd.forward(pv_mv);
  }
  return result;
}

chess::move search_stack::ponder_move() const noexcept { return *(future_.begin()->pv_.begin() + 1); }

search_stack& search_stack::update_selective_depth(const depth_type& height) noexcept {
  selective_depth_ = std::max(selective_depth_, height);
  return *this;
}

search_stack& search_stack::clear_future() noexcept {
  selective_depth_ = 0;
  future_.fill(stack_entry{});
  return *this;
}

search_stack::search_stack(const chess::board_history& past, const chess::board& present) noexcept : past_{past}, present_{present} {}

}  // namespace search
