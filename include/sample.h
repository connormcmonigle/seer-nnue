#pragma once
#include <iomanip>
#include <iostream>
#include <cstdint>

#include <search_util.h>
#include <board.h>

namespace selfplay{


struct sample{
    chess::board bd{};
    search::score_type score;
    int outcome;
};


std::ostream& operator<<(std::ostream& ostr, const sample& x){
  return ostr << std::setprecision(8) << x.bd.fen() << " [" << x.score << "] " << " [" << x.outcome << "]";
}



}
