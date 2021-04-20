#pragma once

#include <iostream>
#include <array>
#include <tuple>
#include <type_traits>
#include <algorithm>
#include <cstdint>
#include <cmath>

#include <enum_util.h>
#include <search_constants.h>
#include <move.h>

namespace prediction{

namespace config{

using value_type = float;
constexpr value_type init_value = static_cast<value_type>(0);

}

config::value_type sigmoid(const config::value_type& x){
  constexpr config::value_type one = static_cast<config::value_type>(1);
  return one / (one + std::exp(-x));
}

config::value_type log_sigmoid(const config::value_type& x){
  constexpr config::value_type one = static_cast<config::value_type>(1);
  return -std::log(one + std::exp(-x));
}

struct sample{
  chess::move present{chess::move::null()};
  chess::move counter{chess::move::null()};
  chess::move follow{chess::move::null()};
};

// present
struct present_from{
  static constexpr size_t size = 64;
  static constexpr bool condition(const sample&){ return true; }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.present.from().index()); }
};


struct present_to{
  static constexpr size_t size = 64;
  static constexpr bool condition(const sample&){ return true; }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.present.to().index()); }
};

struct present_type{
  static constexpr size_t size = 6;
  static constexpr bool condition(const sample&){ return true; }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.present.piece()); }
};

//counter
struct counter_from{
  static constexpr size_t size = 64;
  static constexpr bool condition(const sample& x){ return !x.counter.is_null(); }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.counter.from().index()); }
};


struct counter_to{
  static constexpr size_t size = 64;
  static constexpr bool condition(const sample& x){ return !x.counter.is_null(); }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.counter.to().index()); }
};

struct counter_type{
  static constexpr size_t size = 6;
  static constexpr bool condition(const sample& x){ return !x.counter.is_null(); }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.counter.piece()); }
};

//follow
struct follow_from{
  static constexpr size_t size = 64;
  static constexpr bool condition(const sample& x){ return !x.follow.is_null(); }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.follow.from().index()); }
};


struct follow_to{
  static constexpr size_t size = 64;
  static constexpr bool condition(const sample& x){ return !x.follow.is_null(); }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.follow.to().index()); }
};

struct follow_type{
  static constexpr size_t size = 6;
  static constexpr bool condition(const sample& x){ return !x.follow.is_null(); }
  static constexpr size_t index(const sample& x){ return static_cast<size_t>(x.follow.piece()); }
};


template<typename A0, typename A1, typename ... As>
constexpr size_t calc_index_(const sample& x){ return A0::index(x) + A0::size * calc_index_<A1, As...>(x); }

template<typename A>
constexpr size_t calc_index_(const sample& x){ return A::index(x); }

template<typename ... Ts>
struct times{
  static constexpr size_t size = (Ts::size * ...);
  static constexpr bool condition(const sample& x){ return (Ts::condition(x) && ...); }
  static constexpr size_t index(const sample& x){ return calc_index_<Ts...>(x); }
};


template<typename T>
struct component{
  std::array<config::value_type, T::size> data;
  
  bool condition(const sample& x) const { return T::condition(x); }
  
  const config::value_type& value(const sample& x) const { return data[T::index(x)]; }
  config::value_type& value(const sample& x){ return data[T::index(x)]; }

  component(){ data.fill(config::init_value); }
};


template<typename ... Ts>
struct model{
  static constexpr config::value_type learning_rate = static_cast<config::value_type>(0.1);

  std::tuple<component<Ts>...> components{};

  template<typename T>
  config::value_type compute_value_for(const sample& x) const {
    if(std::get<T>(components).condition(x)){ return std::get<T>(components).value(x); }
    return config::init_value;
  }

  config::value_type predict(const sample& x) const {
    return (compute_value_for<component<Ts>>(x) + ...) / static_cast<config::value_type>(sizeof...(Ts));
  }

  template<typename T>
  void grad_descsent_for(const sample& x, const config::value_type& grad){
    if(std::get<T>(components).condition(x)){ std::get<T>(components).value(x) -= learning_rate * grad; }
  }

  config::value_type add_positive(const sample& x){
    const config::value_type loss = -log_sigmoid(predict(x));
    const config::value_type grad = -sigmoid(-predict(x)) / static_cast<config::value_type>(sizeof...(Ts));
    std::initializer_list<int>{(grad_descsent_for<component<Ts>>(x, grad), 0)...};
    return loss;
  }

  config::value_type add_negative(const sample& x){
    const config::value_type loss = -log_sigmoid(-predict(x));
    const config::value_type grad = sigmoid(predict(x)) / static_cast<config::value_type>(sizeof...(Ts));
    std::initializer_list<int>{(grad_descsent_for<component<Ts>>(x, grad), 0)...};
    return loss;
  }
};

using test_model = model<times<present_from, present_to>, times<counter_from, counter_type, present_to, present_type>, times<follow_from, follow_type, present_to, present_type>>;

//using test_model = model<times<present_from, present_to>, times<counter_to, counter_type, present_to, present_type>>;

//using test_model = model<times<present_from, present_to>>;

}