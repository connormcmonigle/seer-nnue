#pragma once

#include <board.h>
#include <nnue_model.h>
#include <nnue_util.h>
#include <zobrist_util.h>

#include <array>

namespace nnue {

template <typename P>
struct kpt_cache_entry {
  zobrist::hash_type kpt_hash{zobrist::empty_key};
  P encoding{};
};

template <typename T>
struct kpt_cache {
  using p_encoding_type = typename weights<T>::p_encoding_type;
  static constexpr size_t size_mb = 8;
  static constexpr size_t N = (size_mb << 20) / sizeof(kpt_cache_entry<p_encoding_type>);

  const weights<T>* weights_;
  std::array<kpt_cache_entry<p_encoding_type>, N> data{};

  p_encoding_type encoding(const chess::board& bd) {
    auto* entry = &data[bd.kpt_hash() % N];
    if (bd.kpt_hash() == entry->kpt_hash) { return entry->encoding; }
    const p_encoding_type p_encoding = bd.show_pawn_init(p_eval<T>(weights_)).propagate(bd.turn());
    *entry = kpt_cache_entry<p_encoding_type>{bd.kpt_hash(), p_encoding};
    return p_encoding;
  }

  void clear() { data.fill(kpt_cache_entry<p_encoding_type>{}); }

  kpt_cache(const weights<T>* weights) : weights_{weights} {}
};

}  // namespace nnue