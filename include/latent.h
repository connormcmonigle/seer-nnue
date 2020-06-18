#pragma once

#include <array>
#include <limits>
#include <vector>
#include <tuple>

#include <enum_util.h>
#include <square.h>
#include <move.h>
#include <table_generation.h>

namespace chess{

struct latent{
 private:
  bool oo_{true};
  bool ooo_{true};
  square_set ep_mask_{};

 public:

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
    oo_ = val;
    return *this;
  }

  latent& set_ooo(bool val){
    oo_ = val;
    return *this;
  }

  latent& clear_ep_mask(){
    ep_mask_ = square_set{};
    return *this;
  }

  template<typename S>
  latent& set_ep_mask(const S& at){
    static_assert(is_square_v<S>, "at must be of square type");
    clear_ep_mask();
    ep_mask_.add_(at);
    return *this;
  }
};

}