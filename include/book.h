#pragma once

#include <move.h>
#include <zobrist_util.h>

#include <cstdint>
#include <cstring>
#include <fstream>
#include <optional>
#include <sstream>
#include <unordered_map>

namespace chess {

struct book {
  static constexpr char delimiter = '|';
  static constexpr std::string_view dump_path = "dump.bin";

  static constexpr std::uint32_t upper_half(const zobrist::hash_type& hash) {
    return static_cast<std::uint32_t>(hash >> static_cast<std::uint64_t>(32));
  }

  std::unordered_map<std::uint32_t, std::uint32_t> positions{};

  size_t size() const { return positions.size(); }

  std::optional<move> find(const zobrist::hash_type& key) const {
    const auto it = positions.find(upper_half(key));
    if (it != positions.end()) { return std::optional(move(std::get<1>(*it))); }
    return std::nullopt;
  }

  void load(const std::string& path) {
    constexpr std::string_view bin_extension = ".bin";
    constexpr std::string_view txt_extension = ".txt";
    positions.clear();
    if (path.find(std::string(bin_extension)) != std::string::npos) {
      load_binary(path);
    } else if (path.find(std::string(txt_extension)) != std::string::npos) {
      load_plain(path);
    }
  }

  void load_binary(const std::string& path) {
    positions.clear();
    std::fstream reader(path, std::ios_base::in | std::ios_base::binary);

    std::array<char, sizeof(std::uint32_t) + sizeof(std::uint32_t)> single_element{};
    while (reader.read(single_element.data(), single_element.size())) {
      std::uint32_t key{};
      std::uint32_t val{};
      std::memcpy(&key, single_element.data(), sizeof(std::uint32_t));
      std::memcpy(&val, single_element.data() + sizeof(std::uint32_t), sizeof(std::uint32_t));
      positions.insert(std::make_pair(key, val));
    }
  }

  void load_plain(const std::string& path) {
    positions.clear();
    std::fstream reader(path);

    std::string line{};
    while (std::getline(reader, line)) {
      std::stringstream ss(line);
      std::string fen{};
      std::getline(ss, fen, delimiter);
      std::string mv_name{};
      std::getline(ss, mv_name, delimiter);

      const board bd = board::parse_fen(fen);
      const move_list list = bd.generate_moves();
      const auto it = std::find_if(list.begin(), list.end(), [=](const move& mv) { return mv.name(bd.turn()) == mv_name; });

      if (it != list.end()) { positions.insert(std::make_pair(upper_half(bd.hash()), it->data)); }
    }
  }

  void dump() {
    auto writer = std::fstream(std::string(dump_path));
    std::array<char, sizeof(std::uint32_t) + sizeof(std::uint32_t)> single_element{};
    for (const auto& [key, val] : positions) {
      std::memcpy(single_element.data(), &key, sizeof(std::uint32_t));
      std::memcpy(single_element.data() + sizeof(std::uint32_t), &val, sizeof(std::uint32_t));
      writer.write(single_element.data(), single_element.size());
    }
  }
};

}  // namespace chess