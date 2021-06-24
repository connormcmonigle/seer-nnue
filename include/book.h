#pragma once

#include <move.h>
#include <zobrist_util.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <unordered_map>

namespace chess {

struct book {
  std::unordered_map<std::uint32_t, std::uint32_t> positions{};

  size_t size() const { return positions.size(); }

  std::optional<move> find(const zobrist::hash_type& key) const {
    const auto it = positions.find(static_cast<std::uint32_t>(key >> static_cast<std::uint64_t>(32)));
    if (it != positions.end()) { return std::optional(move(std::get<1>(*it))); }
    return std::nullopt;
  }

  void load(const std::string& book_path) {
    positions.clear();
    std::fstream reader(book_path, std::ios_base::in | std::ios_base::binary);

    std::array<char, sizeof(std::uint32_t) + sizeof(std::uint32_t)> single_element{};
    while (reader.read(single_element.data(), single_element.size())) {
      std::uint32_t key{};
      std::uint32_t val{};
      std::memcpy(&key, single_element.data(), sizeof(std::uint32_t));
      std::memcpy(&val, single_element.data() + sizeof(std::uint32_t), sizeof(std::uint32_t));
      positions.insert(std::make_pair(key, val));
    }
  }
};

}  // namespace chess