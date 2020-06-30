#pragma once
#include <iostream>
#include <iterator>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string_view>

#include <bit_field.h>
#include <zobrist_util.h>
#include <move.h>

namespace chess{

enum class bound_type{
  upper,
  lower
};

constexpr std::string_view bound_type_name(const bound_type& type){
  switch(type){
    case bound_type::upper: return "upper";
    case bound_type::lower: return "lower";
    default: return "?";
  }
}

struct tt_entry{
  static constexpr size_t num_bytes = sizeof(tt_entry);
  static constexpr int score_byte_count = sizeof(float);
  static_assert(score_byte_count == 4, "system float type must be 32 bits");

  using type_ = bit_field<bound_type, 0, 2>;
  using score_ = bit_field<std::uint32_t, 2, 34>;
  using best_move_ = bit_field<std::uint32_t, 34, 34+move::width>;

  zobrist::hash_type key_;
  zobrist::hash_type value_;
  int depth_;

  const zobrist::hash_type& key() const { return key_; }
  const zobrist::hash_type& value() const { return value_; }
  int depth() const { return depth_; }

  bound_type bound() const {
    return type_::get(value_);
  }

  float score() const {
    const std::uint32_t raw = score_::get(value_);
    float result; std::memcpy(&result, &raw, score_byte_count);
    return result;
  }

  move best_move() const {
    std::uint32_t mv = best_move_::get(value_);
    return move{mv};
  }

  tt_entry(
    const zobrist::hash_type& key,
    const bound_type type,
    const float score,
    const chess::move& mv,
    const int depth) : key_{key}, value_{0}, depth_{depth}
  {
    type_::set(value_, type);
    std::uint32_t raw; std::memcpy(&raw, &score, score_byte_count);
    score_::set(value_, raw);
    best_move_::set(value_, mv.data);
  }

  tt_entry(const zobrist::hash_type& k, const zobrist::hash_type& v, const int depth) : key_{k}, value_{v}, depth_{depth} {}
  tt_entry() : key_{0}, value_{0}, depth_{0} {}
};

std::ostream& operator<<(std::ostream& ostr, const tt_entry& entry){
  ostr << "tt_entry(key=" << entry.key();
  ostr << ", key^value=" << (entry.key() ^ entry.value());
  ostr << ", best_move=" << entry.best_move();
  ostr << ", bound=" << bound_type_name(entry.bound());
  ostr << ", score=" << entry.score();
  return ostr << ", depth=" << entry.depth() << ')';
}

struct table{
  static constexpr size_t MiB = static_cast<size_t>(1) << static_cast<size_t>(20);
  std::vector<tt_entry> data;

  std::vector<tt_entry>::const_iterator begin() const { return data.cbegin(); }
  std::vector<tt_entry>::const_iterator end() const { return data.cend(); }


  table& insert(const tt_entry& entry){
    const size_t idx = entry.key() % data.size();
    data[idx] = entry;
    data[idx].key_ ^= entry.value();
    return *this;
  }

  std::vector<tt_entry>::const_iterator find(const zobrist::hash_type& key) const {
    const size_t idx = key % data.size();
    std::vector<tt_entry>::const_iterator result = data.cbegin();
    std::advance(result, idx);
    return (key == (result -> key() ^ result -> value())) ? result : data.cend();
  }

  table(size_t size) : data(size * MiB / tt_entry::num_bytes) {}
};

}
