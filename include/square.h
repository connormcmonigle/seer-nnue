#pragma once

#include <iostream>
#include <cmath>
#include <cstdint>
#include <type_traits>
#include <array>


namespace chess{

constexpr std::uint64_t pop_count(std::uint64_t x){
  return static_cast<std::uint64_t>(__builtin_popcountll(x));
}

constexpr std::uint64_t count_trailing_zeros(std::uint64_t x){
  return static_cast<std::uint64_t>(__builtin_ctzll(x));
}

struct square_set;

struct square{
  std::uint64_t data;
  
  constexpr std::uint64_t bit_board() const { return data; }
  constexpr int index() const { return count_trailing_zeros(data); }
  constexpr int file() const { return index() % 8; }
  constexpr int rank() const { return index() / 8; }

  bool operator==(const square& other) const {
    return other.data == data;
  }

  bool operator!=(const square& other) const {
    return !(*this == other);
  }

  std::string name() const {
    const std::array<char, 8> n_of_file = {'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'};
    const std::array<char, 8> n_of_rank = {'1', '2', '3', '4', '5', '6', '7', '8'};
    return std::string("") + n_of_file[file()] + n_of_rank[rank()];
  }

  template<typename I>
  constexpr static square from_index(const I index){
    static_assert(std::is_integral_v<I>, "square index must be of integral type");
    return square(static_cast<std::uint64_t>(1) << static_cast<std::uint64_t>(index));
  }

  constexpr square(std::uint64_t bb) : data{bb} {}
};

std::ostream& operator<<(std::ostream& ostr, const square& sq){
  std::cout << "square(data=" << sq.data << ")\n";
  constexpr std::uint64_t board_hw = 8;
  auto is_set = [sq](std::uint64_t idx){
    return static_cast<bool>((static_cast<std::uint64_t>(1) << idx) & sq.data);
  };
  for(std::uint64_t rank{0}; rank < board_hw; ++rank){
    for(std::uint64_t file{0}; file < board_hw; ++file){
      const std::uint64_t idx = rank * board_hw + file;
      ostr << (is_set(idx) ? '*' : '.') << ' ';
    }
    ostr << '\n';
  }
  return ostr;
}

struct delta{
  int x{0};
  int y{0};
};

struct tbl_square{
  int file{0};
  int rank{0};
  
  constexpr std::uint64_t index() const {
    return static_cast<uint64_t>(rank * 8 + file);
  }

  constexpr std::uint64_t bitboard() const {
    return static_cast<std::uint64_t>(1) << index();
  }

  constexpr bool is_valid() const {
    return 0 <= file && file < 8 && 0 <= rank && rank < 8;
  }

  constexpr std::uint64_t bit_board() const { return static_cast<std::uint64_t>(1) << index(); }

  constexpr tbl_square rotated() const {
    return tbl_square{7 - file, 7 - rank};
  }

  constexpr square to_square() const {
    return square::from_index(index());
  }

  constexpr tbl_square add(delta d) const {
    return tbl_square{
      file + d.x,
      rank + d.y
    };
  }

  template<typename I>
  static constexpr tbl_square from_index(I index){
    return tbl_square{
      static_cast<int>(index) % 8,
      static_cast<int>(index) / 8
    };
  }
  
  constexpr bool operator==(const tbl_square& other) const {
    return (other.rank) == rank && (other.file == file);
  }
  
  constexpr bool operator!=(const tbl_square& other) const {
    return !(*this == other);
  }
  
  static tbl_square from_name(const std::string& name){
    return tbl_square{7 - static_cast<int>(name[0] - 'a'), static_cast<int>(name[1] - '1')};
  }
};

template<typename T>
inline constexpr bool is_square_v = std::is_same_v<T, square> || std::is_same_v<T, tbl_square>;

struct square_set_iterator{
  using difference_type = long;
  using value_type = square;
  using pointer = const square*;
  using reference = const square&;
  using iterator_category = std::output_iterator_tag;

  std::uint64_t original, remaining;
  
  square_set_iterator& operator++(){
    remaining &= (remaining - static_cast<std::uint64_t>(1));
    return *this;
  }
  
  square_set_iterator operator++(int) {
    auto retval = *this;
    ++(*this);
    return retval;
  }
  
  bool operator==(const square_set_iterator& other) const {
    return other.remaining == remaining;
  }
  
  bool operator!=(const square_set_iterator& other) const {
    return !(*this == other);
  }
  
  square operator*() const {
    return square{remaining & ~(remaining - static_cast<std::uint64_t>(1))};
  }

  square_set_iterator(std::uint64_t set) : remaining{set} {}
};

struct square_set{
  static constexpr std::uint64_t one = static_cast<std::uint64_t>(1);
  using iterator = square_set_iterator;
  std::uint64_t data;

  iterator begin() const {
    return square_set_iterator(data);
  }

  iterator end() const {
    return square_set_iterator(static_cast<std::uint64_t>(0));
  }

  constexpr square_set& add_(const tbl_square& tbl_sq){
    data |= one << static_cast<std::uint64_t>(tbl_sq.index());
    return *this;
  }
  
  constexpr square_set& add_(const square& sq){
    data |= sq.bit_board();
    return *this;
  }
  
  constexpr size_t count() const {
    return pop_count(data);
  }

  constexpr size_t any() const {
    return data != 0;
  }

  constexpr square item() const {
    return square{data};
  }

  constexpr square_set& operator|=(const square_set& other){
    data |= other.data;
    return *this;
  }

  constexpr square_set& operator&=(const square_set& other){
    data &= other.data;
    return *this;
  }

  constexpr square_set& operator^=(const square_set& other){
    data ^= other.data;
    return *this;
  }

  constexpr bool is_member(const square& sq) const {
    return 0 != (sq.bit_board() & data);
  } 

  constexpr square_set& operator|=(const std::uint64_t& other){
    data |= other;
    return *this;
  }

  constexpr square_set& operator&=(const std::uint64_t& other){
    data &= other;
    return *this;
  }

  constexpr square_set& operator^=(const std::uint64_t& other){
    data ^= other;
    return *this;
  }

  template<typename I>
  constexpr bool occ(I idx) const {
    static_assert(std::is_integral_v<I>, "idx must be of integral type");
    return static_cast<bool>(data & (one << static_cast<std::uint64_t>(idx)));
  }
  
  constexpr square_set() : data{0} {}
  constexpr square_set(std::uint64_t set) : data{set} {}
};

square_set operator~(const square_set& ss){
  return square_set(~ss.data);
}

square_set operator&(const square_set& a, const square_set& b){
  return square_set(a.data & b.data);
}

square_set operator|(const square_set& a, const square_set& b){
  return square_set(a.data | b.data);
}

square_set operator^(const square_set& a, const square_set& b){
  return square_set(a.data ^ b.data);
}

std::ostream& operator<<(std::ostream& ostr, const square_set& ss){
  std::cout << "square_set(data=" << ss.data << ")\n";
  constexpr std::uint64_t board_hw = 8;
  for(std::uint64_t rank{0}; rank < board_hw; ++rank){
    for(std::uint64_t file{0}; file < board_hw; ++file){
      const std::uint64_t idx = rank * board_hw + file;
      ostr << (ss.occ(idx) ? '*' : '.') << ' ';
    }
    ostr << '\n';
  }
  return ostr;
}

template<typename F>
constexpr void over_all(F&& f){
  for(int i(0); i < 8; ++i){
    for(int j(0); j < 8; ++j){
      f(tbl_square{i, j});
    }
  }
}

template<typename F>
constexpr void over_rank(int rank, F&& f){
  for(auto sq = tbl_square{0, rank}; sq.is_valid(); sq = sq.add(delta{1, 0})){
    f(sq);
  }
}

template<typename F>
constexpr void over_file(int file, F&& f){
  for(auto sq = tbl_square{file, 0}; sq.is_valid(); sq = sq.add(delta{0, 1})){
    f(sq);
  }
}

constexpr square_set gen_rank(int rank){
  square_set ss{};
  over_rank(rank, [&ss](tbl_square& sq){ ss.add_(sq); });
  return ss;
}

constexpr square_set gen_file(int file){
  square_set ss{};
  over_file(file, [&ss](tbl_square& sq){ ss.add_(sq); });
  return ss;
}

}
