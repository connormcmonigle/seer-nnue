#pragma once

#include <cctype>
#include <type_traits>
#include <string_view>

namespace chess{

enum class color{
  white,
  black
};

template<color> struct them_{};

template<>
struct them_<color::white>{
  static constexpr color value = color::black;
};

template<>
struct them_<color::black>{
  static constexpr color value = color::white;
};

template<typename T, typename U>
struct sided{
  using return_type = U;
  
  T& cast(){ return static_cast<T&>(*this); }
  const T& cast() const { return static_cast<const T&>(*this); }
  
  template<color c>
  return_type& us(){
    if constexpr(c == color::white){ return cast().white; }
    else{ return cast().black; }
  }
  
  template<color c>
  const return_type& us() const {
    if constexpr(c == color::white){ return cast().white; }
    else{ return cast().black; }
  }
  
  template<color c>
  return_type& them(){ return us<them_<c>::value>(); }

  template<color c>
  const return_type& them() const { return us<them_<c>::value>(); }
  
  return_type& us(bool side){
    return side ? us<color::white>() : us<color::black>();
  }
  
  const return_type& us(bool side) const {
    return side ? us<color::white>() : us<color::black>();
  }
  
  return_type& them(bool side){
    return us(!side);
  }
  
  const return_type& them(bool side) const {
    return us(!side);
  }
  
  return_type& us(color side){
    return us(side == color::white);
  }
  
  const return_type& us(color side) const {
    return us(side == color::white);
  }
  
  return_type& them(color side){
    return us(side != color::white);
  }
  
  const return_type& them(color side) const {
    return us(side != color::white);
  }
  
  private:
  sided(){};
  friend T;
};

color color_from(char ch){
  return std::isupper(ch) ? color::white : color::black;
}

enum class piece_type{
  pawn,
  knight,
  bishop,
  rook,
  queen,
  king
};

piece_type type_from(char ch){
  switch(std::tolower(ch)){
    case 'p': return piece_type::pawn;
    case 'n': return piece_type::knight;
    case 'b': return piece_type::bishop;
    case 'r': return piece_type::rook;
    case 'q': return piece_type::queen;
    case 'k': return piece_type::king;
    default: return piece_type::king;
  }
}

constexpr std::string_view piece_name(piece_type p){
  switch(p){
    case piece_type::pawn: return "pawn";
    case piece_type::knight: return "knight";
    case piece_type::bishop: return "bishop";
    case piece_type::rook: return "rook";
    case piece_type::queen: return "queen";
    case piece_type::king: return "king";
    default: return "?";
  }
}

template<typename F>
constexpr void over_types(F&& f){
  f(piece_type::king);
  f(piece_type::queen);
  f(piece_type::rook);
  f(piece_type::bishop);
  f(piece_type::knight);
  f(piece_type::pawn);
}

}
