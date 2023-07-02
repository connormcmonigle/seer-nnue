#include <chess/square.h>

#include <array>

namespace chess {

std::string square::name() const noexcept {
  constexpr std::array<char, 8> name_of_file = {'h', 'g', 'f', 'e', 'd', 'c', 'b', 'a'};
  constexpr std::array<char, 8> name_of_rank = {'1', '2', '3', '4', '5', '6', '7', '8'};
  return std::string{} + name_of_file[file()] + name_of_rank[rank()];
}

std::ostream& operator<<(std::ostream& ostr, const square& sq) noexcept {
  ostr << "square(data=" << sq.data << ")\n";

  constexpr square::data_type board_hw = 8;
  auto is_set = [sq](square::data_type idx) { return static_cast<bool>((static_cast<square::data_type>(1) << idx) & sq.data); };

  for (square::data_type rank{0}; rank < board_hw; ++rank) {
    for (square::data_type file{0}; file < board_hw; ++file) {
      const square::data_type idx = rank * board_hw + file;
      ostr << (is_set(idx) ? '*' : '.') << ' ';
    }

    ostr << '\n';
  }

  return ostr;
}

tbl_square tbl_square::from_name(const std::string& name) noexcept {
  return tbl_square{7 - static_cast<int>(name[0] - 'a'), static_cast<int>(name[1] - '1')};
}

std::ostream& operator<<(std::ostream& ostr, const square_set& ss) noexcept {
  std::cout << "square_set(data=" << ss.data << ")\n";
  constexpr square::data_type board_hw = 8;
  for (square::data_type rank{0}; rank < board_hw; ++rank) {
    for (square::data_type file{0}; file < board_hw; ++file) {
      const square::data_type idx = rank * board_hw + file;
      ostr << (ss.occ(idx) ? '*' : '.') << ' ';
    }
    ostr << '\n';
  }
  return ostr;
}

}  // namespace chess