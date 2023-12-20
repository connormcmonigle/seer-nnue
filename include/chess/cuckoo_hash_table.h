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
#include <chess/board_state.h>
#include <chess/square.h>
#include <chess/table_generation.h>
#include <util/bit_range.h>
#include <util/swap.h>
#include <zobrist/util.h>

#include <array>
#include <cstddef>
#include <iostream>
#include <optional>

namespace chess {

struct cuckoo_hash_table_entry {
  using data_type = std::uint16_t;

  using one_ = util::bit_range<std::uint8_t, 0, 6>;
  using two_ = util::next_bit_range<one_, std::uint8_t, 6>;
  using piece_ = util::next_bit_range<two_, piece_type, 3>;

  data_type data;

  [[nodiscard]] constexpr piece_type piece() const noexcept { return piece_::get(data); }
  [[nodiscard]] constexpr square one() const noexcept { return square::from_index(one_::get(data)); }
  [[nodiscard]] constexpr square two() const noexcept { return square::from_index(two_::get(data)); }

  cuckoo_hash_table_entry() = default;

  constexpr cuckoo_hash_table_entry(const square& one, const square& two, const piece_type& piece) : data{} {
    one_::set(data, one.index());
    two_::set(data, two.index());
    piece_::set(data, piece);
  }

  [[nodiscard]] static constexpr cuckoo_hash_table_entry empty() { return cuckoo_hash_table_entry{}; }
};

constexpr bool operator==(const cuckoo_hash_table_entry& a, const cuckoo_hash_table_entry& b) { return a.data == b.data; }

std::ostream& operator<<(std::ostream& ostr, const cuckoo_hash_table_entry& mv) noexcept {
  ostr << "cuckoo_hash_table_entry(one=" << mv.one().name() << ", two=" << mv.two().name() << ", piece=" << piece_name(mv.piece()) << ')';
  return ostr;
}

template <std::size_t N>
struct cuckoo_hash_table_impl {
  static_assert((N != 0) && ((N & (N - 1)) == 0), "N must be a power of 2");

  using value_type = cuckoo_hash_table_entry;
  static constexpr std::size_t mask = N - 1;
  static constexpr zobrist::hash_type initial_hash = zobrist::hash_type{};

  std::array<zobrist::hash_type, N> hashes_{};
  std::array<cuckoo_hash_table_entry, N> entries_{};

  [[nodiscard]] constexpr std::size_t a_hash_function(const zobrist::hash_type& hash) const noexcept { return zobrist::lower_half(hash) & mask; }
  [[nodiscard]] constexpr std::size_t b_hash_function(const zobrist::hash_type& hash) const noexcept { return zobrist::upper_half(hash) & mask; }

  [[nodiscard]] constexpr std::optional<cuckoo_hash_table_entry> look_up(const zobrist::hash_type& hash) const noexcept {
    const std::size_t a_index = a_hash_function(hash);
    const std::size_t b_index = b_hash_function(hash);
    if (hashes_[a_index] == hash) { return entries_[a_index]; }
    if (hashes_[b_index] == hash) { return entries_[b_index]; }
    return std::nullopt;
  }

  constexpr void insert(zobrist::hash_type hash, cuckoo_hash_table_entry entry) noexcept {    
    std::size_t index = a_hash_function(hash);

    for (;;) {
        util::copy_swap(entry, entries_[index]);
        util::copy_swap(hash, hashes_[index]);

        if (hash == initial_hash) { break; }
        
        const std::size_t a_index = a_hash_function(hash);
        const std::size_t b_index = b_hash_function(hash);
        index = (index == a_index) ? b_index : a_index;
    }
  }

  constexpr cuckoo_hash_table_impl() {
    for (auto& elem : hashes_) { elem = initial_hash; }
  }

  static constexpr cuckoo_hash_table_impl<N> make() {
    cuckoo_hash_table_impl<N> result{};

    auto insert_move = [&result](const square& from, const square& to, const piece_type& pt) {
      const zobrist::hash_type white_delta = sided_manifest::w_manifest_src.get(pt, from) ^ sided_manifest::w_manifest_src.get(pt, to);
      const zobrist::hash_type black_delta = sided_manifest::b_manifest_src.get(pt, from) ^ sided_manifest::b_manifest_src.get(pt, to);
      const cuckoo_hash_table_entry entry = cuckoo_hash_table_entry(from, to, pt);

      if (!result.look_up(white_delta).has_value()) { result.insert(white_delta, entry); }
      if (!result.look_up(black_delta).has_value()) { result.insert(black_delta, entry); }
    };

    for (const square from : square_set::all()) {
      const square_set king_attacks = king_attack_tbl.look_up(from);
      const square_set knight_attacks = knight_attack_tbl.look_up(from);
      const square_set bishop_attacks = bishop_attack_tbl.look_up(from, square_set{});
      const square_set rook_attacks = rook_attack_tbl.look_up(from, square_set{});
      const square_set queen_attacks = bishop_attacks | rook_attacks;

      for (const square to : king_attacks) { insert_move(from, to, piece_type::king); }
      for (const square to : knight_attacks) { insert_move(from, to, piece_type::knight); }
      for (const square to : bishop_attacks) { insert_move(from, to, piece_type::bishop); }
      for (const square to : rook_attacks) { insert_move(from, to, piece_type::rook); }
      for (const square to : queen_attacks) { insert_move(from, to, piece_type::queen); }
    }

    return result;
  }
};

struct cuckoo_hash_table {
  using value_type = cuckoo_hash_table_impl<8192>;
  static inline value_type instance = value_type::make();
};

}  // namespace chess