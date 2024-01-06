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

#include <search/move_orderer.h>

#include <algorithm>

namespace search {

void move_orderer_stepper::update_list_() const noexcept {
  auto comparator = [](const move_orderer_entry& a, const move_orderer_entry& b) { return a.sort_key() < b.sort_key(); };
  std::iter_swap(begin_, std::max_element(begin_, end_, comparator));
}

void move_orderer_stepper::next() noexcept {
  ++begin_;
  if (begin_ != end_) { update_list_(); }
}

move_orderer_stepper& move_orderer_stepper::initialize(const move_orderer_data& data, const chess::move_list& list) noexcept {
  const history::context ctxt{data.follow, data.counter, data.threatened, data.pawn_hash};

  end_ = std::transform(list.begin(), list.end(), entries_.begin(), [&data, &ctxt](const chess::move& mv) {
    if (mv.is_noisy()) { return move_orderer_entry::make_noisy(mv, data.bd->see_gt(mv, 0), data.hh->compute_value(ctxt, mv)); }
    return move_orderer_entry::make_quiet(mv, data.killer, data.hh->compute_value(ctxt, mv));
  });

  end_ = std::remove_if(begin_, end_, [&data](const auto& entry) { return entry.mv == data.first; });

  if (begin_ != end_) { update_list_(); }
  is_initialized_ = true;
  return *this;
}

template <typename mode>
std::tuple<int, chess::move> move_orderer_iterator<mode>::operator*() const noexcept {
  if (!stepper_.is_initialized()) { return std::tuple(idx, data_.first); }
  return std::tuple(idx, stepper_.current_move());
}

template <typename mode>
move_orderer_iterator<mode>& move_orderer_iterator<mode>::operator++() noexcept {
  if (!stepper_.is_initialized()) {
    stepper_.initialize(data_, data_.bd->generate_moves<mode>());
  } else {
    stepper_.next();
  }

  ++idx;
  return *this;
}

template <typename mode>
move_orderer_iterator<mode>::move_orderer_iterator(const move_orderer_data& data) noexcept : data_{data} {
  if (data.first.is_null() || !data.bd->is_legal<mode>(data.first)) { stepper_.initialize(data, data.bd->generate_moves<mode>()); }
}

}  // namespace search

template struct search::move_orderer_iterator<chess::generation_mode::all>;
template struct search::move_orderer_iterator<chess::generation_mode::noisy_and_check>;
