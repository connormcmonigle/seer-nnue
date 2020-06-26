#pragma once

#include <iostream>
#include <iterator>
#include <algorithm>

#include <move.h>

namespace chess{

struct move_picker{
  size_t index{0};
  move_list list_;

  bool empty() const {
    return index >= list_.size();
  }

  move peek(){
    auto best = [this](const size_t i0, const size_t i1){
      //ugly heuristics
      const move a = list_.data[i0];
      const move b = list_.data[i1];
      if(a.is_capture() && !b.is_capture()){
        return i0;
      }else if(!a.is_capture() && b.is_capture()){
        return i1;
      }else if(a.is_capture() && b.is_capture()){
        const int victim_a = static_cast<int>(a.captured());
        const int victim_b = static_cast<int>(b.captured());
        if(victim_a > victim_b){
          return i0;
        }else if(victim_a < victim_b){
          return i1;
        }else if(victim_a == victim_b){
          const int type_a = static_cast<int>(a.piece());
          const int type_b = static_cast<int>(b.piece());
          if(type_a < type_b){
            return i0;
          }else{
            return i1;
          }
        }
      }
      //TODO: implement move history heuristic
      return i0;
    };

    size_t best_index = index;
    for(size_t i(index); i < list_.size(); ++i){
      best_index = best(best_index, i);
    }
    std::swap(list_.data[index], list_.data[best_index]);
    return list_.data[index];
  }

  move pick(){
    const move result = peek();
    ++index;
    return result;
  }

  move_picker(const move_list& list) : list_{list} {}

};

}
