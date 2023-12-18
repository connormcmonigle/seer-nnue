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

#include <cstdint>
#include <iostream>
#include <type_traits>
#include <utility>

namespace chess {

[[nodiscard]] constexpr std::uint64_t pop_count(const std::uint64_t& x) noexcept { return static_cast<std::uint64_t>(__builtin_popcountll(x)); }
[[nodiscard]] constexpr std::uint64_t count_trailing_zeros(const std::uint64_t& x) noexcept { return static_cast<std::uint64_t>(__builtin_ctzll(x)); }

struct square_set;

struct square {
  using data_type = std::uint64_t;

  data_type data;

  [[nodiscard]] constexpr const square::data_type& bit_board() const noexcept { return data; }

  [[nodiscard]] constexpr int index() const noexcept { return count_trailing_zeros(data); }
  [[nodiscard]] constexpr int file() const noexcept { return index() % 8; }
  [[nodiscard]] constexpr int rank() const noexcept { return index() / 8; }

  [[nodiscard]] constexpr bool operator==(const square& other) const noexcept { return other.data == data; }
  [[nodiscard]] constexpr bool operator!=(const square& other) const noexcept { return !(*this == other); }

  [[nodiscard]] std::string name() const noexcept;

  constexpr explicit square(const data_type& bb) noexcept : data{bb} {}

  template <typename I>
  [[nodiscard]] constexpr static square from_index(const I& index) noexcept {
    static_assert(std::is_integral_v<I>, "square index must be of integral type");
    return square(static_cast<data_type>(1) << static_cast<data_type>(index));
  }
};

std::ostream& operator<<(std::ostream& ostr, const square& sq) noexcept;

struct delta {
  int x{0};
  int y{0};
};

struct tbl_square {
  int file{0};
  int rank{0};

  [[nodiscard]] constexpr square::data_type index() const noexcept { return static_cast<uint64_t>(rank * 8 + file); }
  [[nodiscard]] constexpr bool is_valid() const noexcept { return 0 <= file && file < 8 && 0 <= rank && rank < 8; }

  [[nodiscard]] constexpr square::data_type bit_board() const noexcept { return static_cast<square::data_type>(1) << index(); }
  [[nodiscard]] constexpr square to_square() const noexcept { return square::from_index(index()); }

  [[nodiscard]] constexpr tbl_square rotated() const noexcept { return tbl_square{7 - file, 7 - rank}; }
  [[nodiscard]] constexpr tbl_square add(delta d) const noexcept { return tbl_square{file + d.x, rank + d.y}; }

  template <typename I>
  [[nodiscard]] static constexpr tbl_square from_index(I index) noexcept {
    return tbl_square{static_cast<int>(index) % 8, static_cast<int>(index) / 8};
  }

  [[nodiscard]] constexpr bool operator==(const tbl_square& other) const noexcept { return (other.rank) == rank && (other.file == file); }
  [[nodiscard]] constexpr bool operator!=(const tbl_square& other) const noexcept { return !(*this == other); }

  [[nodiscard]] static tbl_square from_name(const std::string& name) noexcept;
};

template <typename T>
inline constexpr bool is_square_v = std::is_same_v<T, square> || std::is_same_v<T, tbl_square>;

struct square_set_iterator {
  using value_type = square;
  using pointer = const square*;
  using reference = const square&;
  using iterator_category = std::input_iterator_tag;

  square::data_type remaining;

  [[maybe_unused]] constexpr square_set_iterator& operator++() noexcept {
    remaining &= (remaining - static_cast<square::data_type>(1));
    return *this;
  }

  [[maybe_unused]] constexpr square_set_iterator operator++(int) noexcept {
    auto retval = *this;
    ++(*this);
    return retval;
  }

  [[nodiscard]] constexpr bool operator==(const square_set_iterator& other) const noexcept { return other.remaining == remaining; }
  [[nodiscard]] constexpr bool operator!=(const square_set_iterator& other) const noexcept { return !(*this == other); }

  [[nodiscard]] constexpr square operator*() const { return square{remaining & ~(remaining - static_cast<square::data_type>(1))}; }

  constexpr explicit square_set_iterator(const square::data_type& set) noexcept : remaining{set} {}
};

struct square_set;
[[nodiscard]] constexpr square_set operator~(const square_set& ss) noexcept;
[[nodiscard]] constexpr square_set operator&(const square_set& a, const square_set& b) noexcept;
[[nodiscard]] constexpr square_set operator|(const square_set& a, const square_set& b) noexcept;
[[nodiscard]] constexpr square_set operator^(const square_set& a, const square_set& b) noexcept;

struct square_set {
  static constexpr square::data_type one = static_cast<square::data_type>(1);
  using iterator = square_set_iterator;
  square::data_type data;

  [[nodiscard]] constexpr iterator begin() const noexcept { return square_set_iterator(data); }
  [[nodiscard]] constexpr iterator end() const noexcept { return square_set_iterator(static_cast<square::data_type>(0)); }

  [[nodiscard]] constexpr square_set excluding(const square& sq) const noexcept { return square_set(data & ~sq.bit_board()); }

  [[maybe_unused]] constexpr square_set& insert(const tbl_square& tbl_sq) noexcept {
    data |= one << static_cast<square::data_type>(tbl_sq.index());
    return *this;
  }

  [[maybe_unused]] constexpr square_set& insert(const square& sq) noexcept {
    data |= sq.bit_board();
    return *this;
  }

  [[nodiscard]] constexpr std::size_t count() const noexcept { return pop_count(data); }
  [[nodiscard]] constexpr bool any() const noexcept { return data != 0; }
  [[nodiscard]] constexpr square item() const noexcept { return square{data}; }
  [[nodiscard]] constexpr bool is_member(const square& sq) const noexcept { return 0 != (sq.bit_board() & data); }

  [[maybe_unused]] constexpr square_set& operator|=(const square_set& other) noexcept {
    data |= other.data;
    return *this;
  }

  [[maybe_unused]] constexpr square_set& operator&=(const square_set& other) noexcept {
    data &= other.data;
    return *this;
  }

  [[maybe_unused]] constexpr square_set& operator^=(const square_set& other) noexcept {
    data ^= other.data;
    return *this;
  }

  [[maybe_unused]] constexpr square_set& operator|=(const square::data_type& other) noexcept {
    data |= other;
    return *this;
  }

  [[maybe_unused]] constexpr square_set& operator&=(const square::data_type& other) noexcept {
    data &= other;
    return *this;
  }

  [[maybe_unused]] constexpr square_set& operator^=(const square::data_type& other) noexcept {
    data ^= other;
    return *this;
  }

  [[nodiscard]] constexpr square_set mirrored() const noexcept {
    return square_set{
        (data << 56) | ((data << 40) & static_cast<square::data_type>(0x00ff000000000000)) |
        ((data << 24) & static_cast<square::data_type>(0x0000ff0000000000)) | ((data << 8) & static_cast<square::data_type>(0x000000ff00000000)) |
        ((data >> 8) & static_cast<square::data_type>(0x00000000ff000000)) | ((data >> 24) & static_cast<square::data_type>(0x0000000000ff0000)) |
        ((data >> 40) & static_cast<square::data_type>(0x000000000000ff00)) | (data >> 56)};
  }

  template <typename I>
  [[nodiscard]] constexpr bool occ(const I& idx) const noexcept {
    static_assert(std::is_integral_v<I>, "idx must be of integral type");
    return static_cast<bool>(data & (one << static_cast<square::data_type>(idx)));
  }

  template <typename... Ts>
  [[nodiscard]] static constexpr square_set of(Ts&&... ts) noexcept {
    auto bit_board = [](auto&& sq) { return sq.bit_board(); };
    auto bit_wise_or = [](auto&&... args) { return (args | ...); };
    return square_set(bit_wise_or(bit_board(std::forward<Ts>(ts))...));
  }

  [[nodiscard]] static constexpr square_set all() noexcept { return ~square_set{}; }

  constexpr square_set() noexcept : data{0} {}
  constexpr explicit square_set(const square::data_type& set) noexcept : data{set} {}
};

[[nodiscard]] constexpr square_set operator~(const square_set& ss) noexcept { return square_set(~ss.data); }
[[nodiscard]] constexpr square_set operator&(const square_set& a, const square_set& b) noexcept { return square_set(a.data & b.data); }
[[nodiscard]] constexpr square_set operator|(const square_set& a, const square_set& b) noexcept { return square_set(a.data | b.data); }
[[nodiscard]] constexpr square_set operator^(const square_set& a, const square_set& b) noexcept { return square_set(a.data ^ b.data); }

std::ostream& operator<<(std::ostream& ostr, const square_set& ss) noexcept;

template <typename F>
constexpr void over_all(F&& f) noexcept {
  for (int i(0); i < 8; ++i) {
    for (int j(0); j < 8; ++j) { f(tbl_square{i, j}); }
  }
}

template <typename F>
constexpr void over_rank(const int& rank, F&& f) noexcept {
  for (auto sq = tbl_square{0, rank}; sq.is_valid(); sq = sq.add(delta{1, 0})) { f(sq); }
}

template <typename F>
constexpr void over_file(const int& file, F&& f) noexcept {
  for (auto sq = tbl_square{file, 0}; sq.is_valid(); sq = sq.add(delta{0, 1})) { f(sq); }
}

[[nodiscard]] constexpr square_set generate_rank(const int& rank) noexcept {
  square_set ss{};
  over_rank(rank, [&ss](tbl_square& sq) { ss.insert(sq); });
  return ss;
}

[[nodiscard]] constexpr square_set generate_file(const int& file) noexcept {
  square_set ss{};
  over_file(file, [&ss](tbl_square& sq) { ss.insert(sq); });
  return ss;
}

}  // namespace chess
