#pragma once
#include <iostream>
#include <iterator>
#include <cstdint>
#include <cstring>
#include <vector>
#include <string_view>

#include <bit_field.h>
#include <zobrist_util.h>

namespace chess{

enum class eval_type{
  upper,
  exact,
  lower
};

constexpr std::string_view eval_type_name(const eval_type& eval){
  switch(eval){
    case eval_type::upper: return "upper";
    case eval_type::exact: return "exact";
    case eval_type::lower: return "lower";
    default: return "exact";
  }
}

struct tt_entry{
  static constexpr int score_byte_count = sizeof(float);
  static_assert(score_byte_count == 4, "system float type must be 32 bits");

  using type_ = bit_field<eval_type, 0, 2>;
  using score_ = bit_field<std::uint32_t, 2, 34>;

  zobrist::hash_type key_;
  zobrist::hash_type value_;

  const zobrist::hash_type& key() const { return key_; }
  const zobrist::hash_type& value() const { return value_; }

  eval_type type() const {
    return type_::get(value_);
  }

  float score() const {
    const std::uint32_t raw = score_::get(value_);
    float result; std::memcpy(&result, &raw, score_byte_count);
    return result;
  }

  tt_entry(const zobrist::hash_type& key, const eval_type& type, const float& score) : key_{key}, value_{0} {
    type_::set(value_, type);
    std::uint32_t raw; std::memcpy(&raw, &score, score_byte_count);
    score_::set(value_, raw);
  }

  tt_entry(const zobrist::hash_type& k, const zobrist::hash_type& v) : key_{k}, value_{v} {}
  tt_entry() : key_{0}, value_{0} {}
};

std::ostream& operator<<(std::ostream& ostr, const tt_entry& entry){
  ostr << "tt_entry(key=" << entry.key();
  ostr << ", key^value=" << (entry.key() ^ entry.value());
  ostr << ", type=" << eval_type_name(entry.type());
  return ostr << ", score=" << entry.score() << ')';
}

struct table{
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

  table(size_t size) : data(size) {}
};

}