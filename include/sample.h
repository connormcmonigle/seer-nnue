#pragma once
#include <iomanip>
#include <iostream>
#include <cstdint>


#include <board.h>

namespace selfplay{

template<typename T>
struct sample{
    chess::board bd{};
    T score;
    int outcome;
};

template<typename T>
std::ostream& operator<<(std::ostream& ostr, const sample<T>& x){
  return ostr << std::setprecision(8) << x.bd.fen() << " [" << x.score << "] " << " [" << x.outcome << "]";
}



}
