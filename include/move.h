#pragma once

#include <iostream>
#include <utility>
#include <cstdint>
#include <algorithm>

#include <bit_field.h>
#include <enum_util.h>
#include <square.h>
#include <table_generation.h>

namespace chess{

inline constexpr std::array<piece_type, 4> promotion_types = {
  piece_type::knight,
  piece_type::bishop,
  piece_type::rook,
  piece_type::queen
};

struct move{
  std::uint32_t data{0};

  static constexpr size_t width = 29;
  using from_ = bit_field<std::uint8_t, 0, 6>;
  using to_ = bit_field<std::uint8_t, 6, 12>;
  using piece_ = bit_field<piece_type, 12, 15>;
  using is_capture_ = bit_field<bool, 15, 16>;
  using is_enpassant_ = bit_field<bool, 16, 17>;
  using captured_ = bit_field<piece_type, 17, 20>;
  using enpassant_sq_ = bit_field<std::uint8_t, 20, 26>;
  using promotion_ = bit_field<piece_type, 26, 29>;

  template<typename B>
  constexpr typename B::field_type get_field_() const {
    return B::get(data);
  }

  template<typename B>
  constexpr move& set_field_(const typename B::field_type info){
    B::set(data, info);
    return *this;
  }

  constexpr square from() const { return square::from_index(get_field_<from_>()); }
  constexpr square to() const { return square::from_index(get_field_<to_>()); }
  constexpr piece_type piece() const { return get_field_<piece_>(); }
  constexpr bool is_capture() const { return get_field_<is_capture_>(); }
  constexpr bool is_enpassant() const { return get_field_<is_enpassant_>(); }
  constexpr piece_type captured() const { return get_field_<captured_>(); }
  constexpr square enpassant_sq() const { return square::from_index(get_field_<enpassant_sq_>()); }
  constexpr piece_type promotion() const { return get_field_<promotion_>(); }

  constexpr bool is_null() const { return data == 0; }

  template<color c>
  constexpr bool is_castle_oo() const {
    return piece() == piece_type::king &&
      from() == castle_info<c>.start_king &&
      to() == castle_info<c>.oo_rook;
  }
  
  template<color c>
  constexpr bool is_castle_ooo() const {
    return piece() == piece_type::king &&
      from() == castle_info<c>.start_king &&
      to() == castle_info<c>.ooo_rook;
  }
  
  template<color c>
  constexpr bool is_promotion() const {
    return piece() == piece_type::pawn &&
      pawn_delta<c>::last_rank.is_member(to());
  }
  
  constexpr bool is_promotion() const {
    return is_promotion<color::white>() || is_promotion<color::black>();
  }
  
  template<color c>
  constexpr bool is_pawn_double() const {
    return piece() == piece_type::pawn &&
      pawn_delta<c>::start_rank.is_member(from()) && 
      pawn_delta<c>::double_rank.is_member(to());
  }
  
  constexpr bool is_quiet() const {
    return !is_capture() && !(is_promotion() && piece_type::queen == promotion());
  }
  
  template<color c>
  std::string name() const {
    if(is_castle_oo<c>()){
      return castle_info<c>.start_king.name() + castle_info<c>.after_oo_king.name();
    }else if(is_castle_ooo<c>()){
      return castle_info<c>.start_king.name() + castle_info<c>.after_ooo_king.name();
    }
    std::string base = from().name() + to().name();
    if(is_promotion<c>()){
      return base + piece_letter(promotion());
    }else{
      return base;
    }
  }

  std::string name(bool pov) const {
    return pov ? name<color::white>() : name<color::black>();
  }

  constexpr move() : data{0} {}
  
  constexpr move(std::uint32_t data) : data{data} {}
  
  constexpr move(
    square from,
    square to,
    piece_type piece,
    bool is_capture=false,
    piece_type captured=piece_type::pawn,
    bool is_enpassant=false,
    square enpassant_sq=square::from_index(0),
    piece_type promotion=piece_type::pawn
  ){
    const auto from_idx = static_cast<std::uint8_t>(from.index());
    const auto to_idx = static_cast<std::uint8_t>(to.index());
    const auto ep_sq_idx = static_cast<std::uint8_t>(enpassant_sq.index());
    set_field_<from_>(from_idx).set_field_<to_>(to_idx).set_field_<piece_>(piece).
    set_field_<is_capture_>(is_capture).set_field_<is_enpassant_>(is_enpassant).
    set_field_<captured_>(captured).set_field_<enpassant_sq_>(ep_sq_idx).
    set_field_<promotion_>(promotion);
  }

  constexpr static move null(){ return move{0}; }

};

inline bool operator==(const move& a, const move& b){
  return a.data == b.data;
}

inline bool operator!=(const move& a, const move& b){
  return !(a == b);
}

std::ostream& operator<<(std::ostream& ostr, const move& mv){
  ostr << "move(from=" << mv.from().name() <<
  ", to=" << mv.to().name() << ", piece=" << piece_name(mv.piece()) <<
  ", is_capture=" << mv.is_capture() << ", capture=" << piece_name(mv.captured()) <<
  ", is_enpassant=" << mv.is_enpassant() << ", enpassant_sq=" << mv.enpassant_sq().name() <<
   ", promotion=" << piece_name(mv.promotion()) << ')';
  return ostr;
}

struct move_list{
  static constexpr size_t max_branching_factor = 192;
  using iterator = std::array<move, max_branching_factor>::iterator;
  using const_iterator = std::array<move, max_branching_factor>::const_iterator;
  size_t size_{0};
  std::array<move, max_branching_factor> data{};
  
  iterator begin(){ return data.begin(); }
  iterator end(){ return data.begin() + size_; }
  const_iterator begin() const { return data.cbegin(); }
  const_iterator end() const { return data.cbegin() + size_; }

  bool has(const move& mv) const { return end() != std::find(begin(), end(), mv); }
  size_t size() const { return size_; }
  bool empty() const { return size() == 0; }

  move& operator[](const size_t& idx){ return data[idx]; }
  const move& operator[](const size_t& idx) const { return data[idx]; }

  move_list& add_(move mv){
    constexpr size_t last_idx = max_branching_factor - 1;
    data[size_] = mv;
    ++size_;
    if(size_ > last_idx){ size_ = last_idx; }
    return *this;
  }

  template<typename ... Ts>
  move_list& add_(const Ts& ... ts){
    return add_(move(ts...));
  }
  
  template<bool gen_quiet, typename ... Ts>
  move_list& add_promotion_(const Ts& ... ts){
    assert((move(ts...).piece() == piece_type::pawn));
    if constexpr(gen_quiet){
      for(const auto& pt : promotion_types){ add_(move(ts...).set_field_<move::promotion_>(pt)); }
    }else{
      add_(move(ts...).set_field_<move::promotion_>(piece_type::queen));
    }
    return *this;
  }
};

std::ostream& operator<<(std::ostream& ostr, const move_list& mv_ls){
  for(size_t i(0); i < mv_ls.size(); ++i){
    ostr << i << ". " << mv_ls[i] << '\n';
  }
  return ostr;
}

}
