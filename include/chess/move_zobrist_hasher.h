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

#include <chess/move.h>
#include <zobrist/util.h>

#include <array>

namespace chess {

struct move_zobrist_hasher_impl {
  static constexpr std::size_t num_pieces = 6;
  static constexpr std::size_t num_squares = 64;

  std::array<zobrist::hash_type, num_pieces> piece_{};
  std::array<zobrist::hash_type, num_squares> to_{};

  constexpr zobrist::hash_type compute_hash(const move& mv) const noexcept {
    const auto piece_index = static_cast<std::size_t>(mv.piece());
    const auto to_index = static_cast<std::size_t>(mv.to().index());

    return piece_[piece_index] ^ to_[to_index];
  }

  constexpr move_zobrist_hasher_impl(zobrist::xorshift_generator generator) noexcept {
    for (auto& elem : piece_) { elem = generator.next(); }
    for (auto& elem : to_) { elem = generator.next(); }
  }
};

template <zobrist::hash_type entropy>
inline constexpr move_zobrist_hasher_impl move_zobrist_hasher = move_zobrist_hasher_impl(zobrist::xorshift_generator(entropy));

constexpr move_zobrist_hasher_impl counter_move_zobrist_hasher = move_zobrist_hasher<zobrist::entropy_0>;
constexpr move_zobrist_hasher_impl follow_move_zobrist_hasher = move_zobrist_hasher<zobrist::entropy_1>;

}  // namespace chess
