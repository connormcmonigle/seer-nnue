#pragma once

#include <search_constants.h>

#include <algorithm>

namespace search {

struct fractional_type {
  static constexpr depth_type numerator = 3;
  static constexpr depth_type denominator = 5;
  depth_type value_;

  constexpr operator depth_type() const { return value_ * numerator / denominator; }
  constexpr depth_type depth() const { return value_ * numerator / denominator; }

  constexpr fractional_type& operator+=(const depth_type& other) {
    value_ = (value_ * numerator + other * denominator) / numerator;
    return *this;
  }

  constexpr fractional_type& operator-=(const depth_type& other) {
    value_ = (value_ * numerator - other * denominator) / numerator;
    return *this;
  }

  fractional_type max(const depth_type& depth) { return fractional_type(std::max(denominator * depth / numerator, value_)); }

  explicit constexpr fractional_type(const depth_type& value) : value_{value} {}

  static constexpr fractional_type from_value(const depth_type& depth) { return fractional_type(depth * denominator / numerator); }
};

constexpr fractional_type operator+(const fractional_type& a, const depth_type& b) {
  return fractional_type((a.value_ * fractional_type::numerator + b * fractional_type::denominator) / fractional_type::numerator);
}

constexpr fractional_type operator+(const depth_type& b, const fractional_type& a) { return a + b; }

constexpr fractional_type operator-(const fractional_type& a, const depth_type& b) {
  return fractional_type((a.value_ * fractional_type::numerator - b * fractional_type::denominator) / fractional_type::numerator);
}

constexpr fractional_type operator-(const depth_type& b, const fractional_type& a) { return a - b; }

}  // namespace search