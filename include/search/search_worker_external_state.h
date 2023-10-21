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

#include <nnue/eval.h>
#include <search/search_constants.h>
#include <search/transposition_table.h>

#include <functional>
#include <memory>

namespace search {
struct search_worker;

struct search_worker_external_state {
  const nnue::quantized_weights* weights;
  std::shared_ptr<transposition_table> tt;
  std::shared_ptr<search_constants> constants;
  std::function<void(const search_worker&)> on_iter;
  std::function<void(const search_worker&)> on_update;

  search_worker_external_state(
      const nnue::quantized_weights* weights_,
      std::shared_ptr<transposition_table> tt_,
      std::shared_ptr<search_constants> constants_,
      std::function<void(const search_worker&)>& on_iter_,
      std::function<void(const search_worker&)> on_update_) noexcept
      : weights{weights_}, tt{tt_}, constants{constants_}, on_iter{on_iter_}, on_update{on_update_} {}
};

}  // namespace search