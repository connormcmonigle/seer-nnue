#pragma once

#include <array>
#include <optional>

#include <zobrist_util.h>
#include <search_constants.h>

namespace chess{

struct eval_cache_entry{
  zobrist::hash_type hash{};
  search::score_type eval{};
};

struct eval_cache{
  static constexpr size_t size_mb = 4;
  static constexpr size_t N = (size_mb << 20) / sizeof(eval_cache_entry);
  
  std::array<eval_cache_entry, N> data{};
  
  std::optional<search::score_type> find(const zobrist::hash_type& hash) const { 
    if(data[hash % N].hash == hash){ return data[hash % N].eval; }
    return std::nullopt;
  }

  void insert(const zobrist::hash_type& hash, const search::score_type& eval){
    data[hash % N] = eval_cache_entry{hash, eval};
  }

  void clear(){ return data.fill(eval_cache_entry{}); }
};

}