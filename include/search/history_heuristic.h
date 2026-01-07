/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021-2023  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#pragma once

#include <chess/move.h>
#include <chess/types.h>
#include <search/search_constants.h>
#include <util/tuple.h>
#include <zobrist/util.h>

#include <algorithm>
#include <array>
#include <cstdint>

namespace search {

namespace history {

using value_type = search::counter_type;
using storage_type = std::int16_t;

namespace constants {

constexpr std::size_t num_squares = 64;
constexpr std::size_t num_pieces = 6;
constexpr std::size_t num_threat_states = 2;

constexpr std::size_t num_pawn_states = 512;
constexpr std::size_t pawn_hash_mask = num_pawn_states - 1;
static_assert((num_pawn_states & pawn_hash_mask) == 0);

constexpr std::size_t num_eval_feature_states = 512;
constexpr std::size_t eval_feature_hash_mask = num_eval_feature_states - 1;
static_assert((num_eval_feature_states & eval_feature_hash_mask) == 0);

constexpr value_type min_storage_limit = std::numeric_limits<storage_type>::min();
constexpr value_type max_storage_limit = std::numeric_limits<storage_type>::max();

}  // namespace constants

struct context {
  chess::move follow;
  chess::move counter;
  chess::square_set threatened;
  zobrist::hash_type pawn_hash;
  zobrist::quarter_hash_type eval_feature_hash;
};

[[nodiscard]] inline value_type formula(const value_type& x, const value_type& gain) noexcept {
  constexpr value_type history_multiplier = 32;
  constexpr value_type history_divisor = 512;
  constexpr value_type history_limit = 16384;

  const value_type clamped_x = std::clamp(x, -history_limit, history_limit);
  return (gain * history_multiplier) - (clamped_x * std::abs(gain) / history_divisor);
}

struct eval_feature_info {
  static constexpr std::size_t N = constants::num_eval_feature_states * constants::num_pieces * constants::num_squares;

  [[nodiscard]] static constexpr bool is_applicable(const context& ctxt, const chess::move& mv) noexcept {
    return ctxt.eval_feature_hash && mv.is_quiet();
  }

  [[nodiscard]] static constexpr std::size_t compute_index(const context& ctxt, const chess::move& mv) noexcept {
    const auto eval_features = static_cast<std::size_t>(ctxt.eval_feature_hash & constants::eval_feature_hash_mask);
    const auto p = static_cast<std::size_t>(mv.piece());
    const auto to = static_cast<std::size_t>(mv.to().index());

    return eval_features * constants::num_pieces * constants::num_squares + p * constants::num_squares + to;
  }
};

struct pawn_structure_info {
  static constexpr std::size_t N = constants::num_pawn_states * constants::num_pieces * constants::num_squares;

  [[nodiscard]] static constexpr bool is_applicable(const context& ctxt, const chess::move& mv) noexcept { return ctxt.pawn_hash && mv.is_quiet(); }

  [[nodiscard]] static constexpr std::size_t compute_index(const context& ctxt, const chess::move& mv) noexcept {
    const auto pawns = static_cast<std::size_t>(ctxt.pawn_hash & constants::pawn_hash_mask);
    const auto p = static_cast<std::size_t>(mv.piece());
    const auto to = static_cast<std::size_t>(mv.to().index());

    return pawns * constants::num_pieces * constants::num_squares + p * constants::num_squares + to;
  }
};

struct threat_info {
  static constexpr std::size_t N = constants::num_threat_states * constants::num_squares * constants::num_squares;

  [[nodiscard]] static constexpr bool is_applicable(const context&, const chess::move& mv) noexcept { return mv.is_quiet(); }

  [[nodiscard]] static constexpr std::size_t compute_index(const context& ctxt, const chess::move& mv) noexcept {
    const auto t = static_cast<std::size_t>(ctxt.threatened.is_member(mv.from()));
    const auto from = static_cast<std::size_t>(mv.from().index());
    const auto to = static_cast<std::size_t>(mv.to().index());

    return t * constants::num_squares * constants::num_squares + from * constants::num_squares + to;
  }
};

struct counter_info {
  static constexpr std::size_t N = constants::num_squares * constants::num_pieces * constants::num_squares * constants::num_pieces;

  [[nodiscard]] static constexpr bool is_applicable(const context& ctxt, const chess::move& mv) noexcept {
    return !ctxt.counter.is_null() && mv.is_quiet();
  }

  [[nodiscard]] static constexpr std::size_t compute_index(const context& ctxt, const chess::move& mv) noexcept {
    const auto p0 = static_cast<std::size_t>(ctxt.counter.piece());
    const auto to0 = static_cast<std::size_t>(ctxt.counter.to().index());
    const auto p1 = static_cast<std::size_t>(mv.piece());
    const auto to1 = static_cast<std::size_t>(mv.to().index());
    return p0 * constants::num_squares * constants::num_pieces * constants::num_squares + to0 * constants::num_pieces * constants::num_squares +
           p1 * constants::num_squares + to1;
  }
};

struct follow_info {
  static constexpr std::size_t N = constants::num_squares * constants::num_pieces * constants::num_squares * constants::num_pieces;

  [[nodiscard]] static constexpr bool is_applicable(const context& ctxt, const chess::move& mv) noexcept {
    return !ctxt.follow.is_null() && mv.is_quiet();
  }

  [[nodiscard]] static constexpr std::size_t compute_index(const context& ctxt, const chess::move& mv) noexcept {
    const auto p0 = static_cast<std::size_t>(ctxt.follow.piece());
    const auto to0 = static_cast<std::size_t>(ctxt.follow.to().index());
    const auto p1 = static_cast<std::size_t>(mv.piece());
    const auto to1 = static_cast<std::size_t>(mv.to().index());
    return p0 * constants::num_squares * constants::num_pieces * constants::num_squares + to0 * constants::num_pieces * constants::num_squares +
           p1 * constants::num_squares + to1;
  }
};

struct capture_info {
  static constexpr std::size_t N = constants::num_squares * constants::num_pieces * constants::num_pieces;

  [[nodiscard]] static constexpr bool is_applicable(const context&, const chess::move& mv) noexcept { return mv.is_capture(); }

  [[nodiscard]] static constexpr std::size_t compute_index(const context&, const chess::move& mv) noexcept {
    const auto piece = static_cast<std::size_t>(mv.piece());
    const auto to = static_cast<std::size_t>(mv.to().index());
    const auto capture = static_cast<std::size_t>(mv.captured());
    return piece * constants::num_squares * constants::num_pieces + to * constants::num_pieces + capture;
  }
};

template <typename T>
struct table {
  std::array<storage_type, T::N> data_{};

  [[nodiscard]] constexpr bool is_applicable(const context& ctxt, const chess::move& mv) const noexcept { return T::is_applicable(ctxt, mv); }

  [[nodiscard]] constexpr const storage_type& at(const context& ctxt, const chess::move& mv) const noexcept {
    return data_[T::compute_index(ctxt, mv)];
  }

  [[nodiscard]] constexpr storage_type& at(const context& ctxt, const chess::move& mv) noexcept { return data_[T::compute_index(ctxt, mv)]; }

  constexpr void clear() noexcept { data_.fill(storage_type{}); }
};

template <typename... Ts>
struct combined {
  std::tuple<table<Ts>...> tables_{};

  [[maybe_unused]] constexpr combined<Ts...>&
  update(const context& ctxt, const chess::move& best_move, const chess::move_list& tried, const depth_type& depth) noexcept {
    constexpr value_type history_max = 400;

    auto single_update = [&, this](const auto& mv, const value_type& gain) {
      const value_type value = compute_value(ctxt, mv);
      const value_type delta = formula(value, gain);

      util::tuple::for_each(tables_, [=](auto& tbl) {
        if (tbl.is_applicable(ctxt, mv)) {
          const value_type updated_value = delta + tbl.at(ctxt, mv);
          const value_type clamped_value = std::clamp(updated_value, constants::min_storage_limit, constants::max_storage_limit);

          tbl.at(ctxt, mv) = static_cast<storage_type>(clamped_value);
        }
      });
    };

    const value_type gain = std::min(history_max, depth * depth);
    std::for_each(tried.begin(), tried.end(), [single_update, gain](const chess::move& mv) { single_update(mv, -gain); });
    single_update(best_move, gain);

    return *this;
  }

  constexpr void clear() noexcept {
    util::tuple::for_each(tables_, [](auto& tbl) { tbl.clear(); });
  }

  [[nodiscard]] constexpr value_type compute_value(const context& ctxt, const chess::move& mv) const noexcept {
    value_type result{};
    util::tuple::for_each(tables_, [&](const auto& tbl) {
      if (tbl.is_applicable(ctxt, mv)) { result += tbl.at(ctxt, mv); }
    });
    return result;
  }
};

}  // namespace history

using history_heuristic = history::combined<
    history::threat_info,
    /*history::pawn_structure_info,*/
    history::eval_feature_info,
    history::counter_info,
    history::follow_info,
    history::capture_info>;

struct sided_history_heuristic : public chess::sided<sided_history_heuristic, history_heuristic> {
  history_heuristic white;
  history_heuristic black;

  constexpr void clear() noexcept {
    white.clear();
    black.clear();
  }

  sided_history_heuristic() noexcept : white{}, black{} {}
};

}  // namespace search
