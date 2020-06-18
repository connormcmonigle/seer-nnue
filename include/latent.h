#pragma once

#include <array>
#include <limits>
#include <algorithm>
#include <tuple>

#include <zobrist_util.h>
#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>

namespace chess{

struct latent_zobrist_src{
  static constexpr size_t num_squares = 64;
  zobrist::hash_type oo_;
  zobrist::hash_type ooo_;
  std::array<zobrist::hash_type, num_squares> ep_mask_;

  zobrist::hash_type get_oo() const { return oo_; }
  zobrist::hash_type get_ooo() const { return ooo_; }
 
  template<typename S>
  zobrist::hash_type get_ep_mask(const S& at) const {
    static_assert(is_square_v<S>, "at must be of square type");
    return ep_mask_[at.index()];
  }

  latent_zobrist_src(){
    oo_ = zobrist::random_bit_string();
    ooo_ = zobrist::random_bit_string();
    std::transform(ep_mask_.begin(), ep_mask_.begin(), ep_mask_.end(), [](auto...){
      return zobrist::random_bit_string();
    });
  }
};

inline const latent_zobrist_src w_latent_src{};
inline const latent_zobrist_src b_latent_src{};

struct latent{
  const latent_zobrist_src* zobrist_src_;
  zobrist::hash_type hash_{};
  bool oo_{true};
  bool ooo_{true};
  square_set ep_mask_{};

  zobrist::hash_type hash() const {
    return hash_;
  }

  bool oo() const {
    return oo_;
  }

  bool ooo() const {
    return ooo_;
  }

  const square_set& ep_mask() const {
    return ep_mask_;
  }

  latent& set_oo(bool val){
    if(val ^ oo_){ hash_ ^= zobrist_src_ -> get_oo(); }
    oo_ = val;
    return *this;
  }

  latent& set_ooo(bool val){
    if(val ^ ooo_){ hash_ ^= zobrist_src_ -> get_ooo(); }
    oo_ = val;
    return *this;
  }

  latent& clear_ep_mask(){
    if(ep_mask_.any()){ hash_ ^= zobrist_src_ -> get_ep_mask(ep_mask_.item()); }
    ep_mask_ = square_set{};
    return *this;
  }

  template<typename S>
  latent& set_ep_mask(const S& at){
    static_assert(is_square_v<S>, "at must be of square type");
    clear_ep_mask();
    hash_ ^= zobrist_src_ -> get_ep_mask(at);
    ep_mask_.add_(at);
    return *this;
  }

  latent(const latent_zobrist_src* src) : zobrist_src_{src} {}
};

}