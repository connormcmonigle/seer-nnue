#pragma once

#include <iostream>
#include <utility>
#include <algorithm>

#include <weights_streamer.h>

namespace nnue{

template<typename T>
constexpr T relu(const T& x){ return std::max(x, T{0}); }

template<typename T, size_t dim>
struct stack_vector{
  T data[dim];
  
  template<typename F>
  constexpr stack_vector<T, dim> apply(F&& f) const {
    return stack_vector<T, dim>{*this}.apply_(std::forward<F>(f));
  }
  
  template<typename F>
  constexpr stack_vector<T, dim>& apply_(F&& f){
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      data[i] = f(data[i]);
    }
    return *this;
  }

  constexpr stack_vector<T, dim>& add_(const T* other){
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      data[i] += other[i];
    }
    return *this;
  }
  
  constexpr stack_vector<T, dim>& sub_(const T* other){
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      data[i] -= other[i];
    }
    return *this;
  }
  
  constexpr stack_vector<T, dim>& fma_(const T c, const T* other){
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      data[i] += c * other[i];
    }
    return *this;
  }
  
  constexpr stack_vector<T, dim>& set_(const T* other){
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      data[i] = other[i];
    }
    return *this;
  }

  constexpr T item() const {
    static_assert(dim == 1, "called item() on vector with dim != 1");
    return data[0];
  }
  
  static constexpr stack_vector<T, dim> zeros(){
    stack_vector<T, dim> result{};
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      result.data[i] = T(0);
    }
    return result;
  }
  
  static constexpr stack_vector<T, dim> ones(){
    stack_vector<T, dim> result{};
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      result.data[i] = T(1);
    }
    return result;
  }
  
  static constexpr stack_vector<T, dim> from(const T* data){
    stack_vector<T, dim> result{};
    #pragma omp simd
    for(size_t i = 0; i < dim; ++i){
      result.data[i] = data[i];
    }
    return result;
  }
};

template<typename T, size_t dim>
std::ostream& operator<<(std::ostream& ostr, const stack_vector<T, dim>& vec){
  static_assert(dim != 0, "can't stream empty vector.");
  ostr << "stack_vector<T, " << dim << ">([";
  for(size_t i = 0; i < (dim-1); ++i){
    ostr << vec.data[i] << ", ";
  }
  ostr << vec.data[dim-1] << "])";
  return ostr;
}

template<typename T, size_t dim0, size_t dim1>
constexpr stack_vector<T, dim0 + dim1> splice(const stack_vector<T, dim0>& a, const stack_vector<T, dim1>& b){
  auto c = stack_vector<T, dim0 + dim1>::zeros();
  #pragma omp simd
  for(size_t i = 0; i < dim0; ++i){
    c.data[i] = a.data[i];
  }
  for(size_t i = 0; i < dim1; ++i){
    c.data[dim0 + i] = b.data[i];
  }
  return c;
}


template<typename T, size_t dim0, size_t dim1>
struct stack_affine{
  static constexpr size_t W_numel = dim0*dim1;
  static constexpr size_t b_numel = dim1;
  
  T W[W_numel];
  T b[b_numel];
  
  constexpr size_t num_parameters() const {
    return W_numel + b_numel;
  }
  
  constexpr stack_vector<T, dim1> forward(const stack_vector<T, dim0>& x) const {
    auto result = stack_vector<T, dim1>::from(b);
    #pragma omp simd
    for(size_t i = 0; i < dim0; ++i){
      result.fma_(x.data[i], W + i * dim1);
    }
    return result;
  }
  
  constexpr stack_vector<T, dim1> relu_forward(const stack_vector<T, dim0>& x) const {
    auto result = stack_vector<T, dim1>::from(b);
    for(size_t i = 0; i < dim0; ++i){
      if(x.data[i] > T{0}){
        result.fma_(x.data[i], W + i * dim1);
      }
    }
    return result;
  }
  
  stack_affine<T, dim0, dim1>& load_(weights_streamer<T>& ws){
    ws.stream(W, W_numel).stream(b, b_numel);
    return *this;
  }
};

template<typename T, size_t dim0, size_t dim1>
struct big_affine{
  static constexpr size_t W_numel = dim0*dim1;
  static constexpr size_t b_numel = dim1;

  T* W{nullptr};
  T b[b_numel];

  constexpr size_t num_parameters() const {
    return W_numel + b_numel;
  }

  void insert_idx(const size_t idx, stack_vector<T, b_numel>& x) const {
    const T* mem_region = W + idx * dim1;
    x.add_(mem_region);
  }
  
  void erase_idx(const size_t idx, stack_vector<T, b_numel>& x) const {
    const T* mem_region = W + idx * dim1;
    x.sub_(mem_region);
  }

  big_affine<T, dim0, dim1>& load_(weights_streamer<T>& ws){
    ws.stream(W, W_numel).stream(b, b_numel);
    return *this;
  }

  big_affine<T, dim0, dim1>& operator=(const big_affine<T, dim0, dim1>& other){
    #pragma omp simd
    for(size_t i = 0; i < W_numel; ++i){ W[i] = other.W[i]; }
    for(size_t i = 0; i < b_numel; ++i){ b[i] = other.b[i]; }
    return *this;
  }

  big_affine<T, dim0, dim1>& operator=(big_affine<T, dim0, dim1>&& other){
    std::swap(W, other.W);
    std::swap(b, other.b);
    return *this;
  }

  big_affine(const big_affine<T, dim0, dim1>& other){
    W = new T[W_numel];
    #pragma omp simd
    for(size_t i = 0; i < W_numel; ++i){ W[i] = other.W[i]; }
    for(size_t i = 0; i < b_numel; ++i){ b[i] = other.b[i]; }
  }

  big_affine(big_affine<T, dim0, dim1>&& other){
    std::swap(W, other.W);
    std::swap(b, other.b);
  }
  
  big_affine(){ W = new T[W_numel]; }
  ~big_affine(){ if(W != nullptr){ delete[] W; } }

};

}
