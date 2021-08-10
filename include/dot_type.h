#pragma once
#include <cstdint>

namespace util {

template<typename T>
struct dot_type_{ using type = T; };

template<>
struct dot_type_<std::int8_t> { using type = std::int16_t; };

template<>
struct dot_type_<std::int16_t> { using type = std::int32_t; };

template<>
struct dot_type_<std::int32_t> { using type = std::int64_t; };

template<typename T>
using dot_type = typename dot_type_<T>::type;

}