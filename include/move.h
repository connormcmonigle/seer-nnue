#pragma once

#include <iostream>
#include <utility>
#include <cstdint>

#include <enum_util.h>
#include <square.h>
#include <table_generation.h>

namespace chess{

template<typename T, size_t B0, size_t B1>
struct bit_field{
  static_assert(B0 < B1, "wrong bit order");
  using field_type = T;
  static constexpr size_t first = B0;
  static constexpr size_t last = B1;
  
  template<typename I>
  static constexpr T get(const I i){
    constexpr I one = static_cast<I>(1);
    constexpr I b0 = static_cast<I>(first);
    constexpr I b1 = static_cast<I>(last);
    constexpr I mask = ((one << (b1 - b0)) - one) << b0;
    return static_cast<T>((mask & i) >> b0);
  }
  
  template<typename I>
  static void set(I& i, const T info){
    constexpr I one = static_cast<I>(1);
    constexpr I b0 = static_cast<I>(first);
    constexpr I b1 = static_cast<I>(last);
    const I info_ = static_cast<I>(info);
    constexpr I mask = ((one << (b1 - b0)) - one) << b0;
    i &= ~mask;
    i |= info_ << b0;
  }
};

struct move{
  std::uint32_t data{0};

  using from_ = bit_field<std::uint8_t, 0, 6>;
  using to_ = bit_field<std::uint8_t, 6, 12>;
  using piece_ = bit_field<piece_type, 12, 16>;
  using is_capture_ = bit_field<bool, 16, 17>;
  using is_enpassant_ = bit_field<bool, 17, 18>;
  using captured_ = bit_field<piece_type, 18, 22>;
  using enpassant_sq_ = bit_field<std::uint8_t, 22, 28>;

  template<typename B>
  typename B::field_type _get_field() const {
    return B::get(data);
  }

  template<typename B>
  move& _set_field(const typename B::field_type info){
    B::set(data, info);
    return *this;
  }

  square from() const { return square::from_index(_get_field<from_>()); }
  square to() const { return square::from_index(_get_field<to_>()); }
  piece_type piece() const { return _get_field<piece_>(); }
  bool is_capture() const { return _get_field<is_capture_>(); }
  bool is_enpassant() const { return _get_field<is_enpassant_>(); }
  piece_type captured() const { return _get_field<captured_>(); }
  square enpassant_sq() const { return square::from_index(_get_field<enpassant_sq_>()); }

  template<color c>
  bool is_castle_oo() const {
    return piece() == piece_type::king &&
      from() == castle_info<c>.start_king &&
      to() == castle_info<c>.oo_rook;
  }
  
  template<color c>
  bool is_castle_ooo() const {
    return piece() == piece_type::king &&
      from() == castle_info<c>.start_king &&
      to() == castle_info<c>.ooo_rook;
  }
  
  template<color c>
  bool is_promotion() const {
    return piece() == piece_type::pawn &&
      pawn_delta<c>::last_rank.is_member(to());
  }
  
  template<color c>
  bool is_pawn_double() const {
    return piece() == piece_type::pawn &&
      pawn_delta<c>::start_rank.is_member(from()) && 
      pawn_delta<c>::double_rank.is_member(to());
  }
  
  std::string name() const {
    return from().name() + to().name();
  }

  move() : data{0} {}
  
  move(std::uint32_t data) : data{data} {}
  
  move(
    square from,
    square to,
    piece_type piece,
    bool is_capture=false,
    piece_type captured=piece_type::pawn,
    bool is_enpassant=false,
    square enpassant_sq=square{0}
  ){
    const auto from_idx = static_cast<std::uint8_t>(from.index());
    const auto to_idx = static_cast<std::uint8_t>(to.index());
    const auto ep_sq_idx = static_cast<std::uint8_t>(enpassant_sq.index());
    _set_field<from_>(from_idx)._set_field<to_>(to_idx)._set_field<piece_>(piece).
    _set_field<is_capture_>(is_capture)._set_field<is_enpassant_>(is_enpassant).
    _set_field<captured_>(captured)._set_field<enpassant_sq_>(ep_sq_idx);
  }

};

std::ostream& operator<<(std::ostream& ostr, const move& mv){
  ostr << "move(from=" << mv.from().name() <<
  ", to=" << mv.to().name() << ", piece=" << piece_name(mv.piece()) <<
  ", is_capture=" << mv.is_capture() << ", capture=" << piece_name(mv.captured()) <<
  ", is_enpassant=" << mv.is_enpassant() << ", enpassant_sq=" << mv.enpassant_sq().name() << ')';
  return ostr;
}

struct move_list{
  static constexpr size_t max_branching_factor = 192;
  size_t size_{0};
  std::array<move, max_branching_factor> data{};
  
  size_t size() const {
    return size_;
  }
  
  move_list& add_(move mv){
    data[size_] = mv;
    ++size_;
    return *this;
  }
  
  template<typename ... Ts>
  move_list& add_(const Ts& ... ts){
    return add_(move(ts...));
  }
};

std::ostream& operator<<(std::ostream& ostr, const move_list& mv_ls){
  for(size_t i(0); i < mv_ls.size(); ++i){
    ostr << i << ". " << mv_ls.data[i] << '\n';
  }
  return ostr;
}

/*std::vector<move> parse_uci_moves(const std::string& move_str){
  std::stringstream ss(move_str);
  std::vector<move> result{};
  std::transform(
    std::istream_iterator<std::string>(ss),
    std::istream_iterator<std::string>(),
    std::back_inserter(result),
    [](const std::string& m){ return move::from_uci(m); }
  );
  return result;
}*/

}
