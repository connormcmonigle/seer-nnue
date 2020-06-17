#include <table_generation.h>

int main(){
  //chess::_rook_attack_tbl(1);
  {
    //for(auto ss : chess::rook_attack_tbl.data){
      //std::cout << ss << std::endl;
    //}
  }
  
  {
    std::cout << "test0:" << std::endl;
    const auto from = chess::square::from_index(chess::tbl_square({3, 4}).index());
    chess::square_set bl{};
    bl.add_(chess::tbl_square{1, 4});
    bl.add_(chess::tbl_square{6, 4});
    bl.add_(chess::tbl_square{3, 6});
    bl.add_(chess::tbl_square{3, 1});
    std::cout << bl << std::endl;
    std::cout << from << std::endl;
    std::cout << chess::rook_attack_tbl.look_up(from, bl) << std::endl;
  }
  
  {
    std::cout << "test1:" << std::endl;
    const auto from = chess::square::from_index(chess::tbl_square({3, 4}).index());
    chess::square_set bl{};
    bl.add_(chess::tbl_square{1, 2});
    bl.add_(chess::tbl_square{5, 6});
    std::cout << bl << std::endl;
    std::cout << from << std::endl;
    std::cout << chess::bishop_attack_tbl.look_up(from, bl) << std::endl;
  }

  {
    std::cout << "test3:" << std::endl;
    const auto from = chess::square::from_index(chess::tbl_square({1, 1}).index());
    chess::square_set bl{};
    bl.add_(chess::tbl_square{1, 3});
    std::cout << bl << std::endl;
    std::cout << from << std::endl;
    std::cout << chess::pawn_push_tbl<chess::color::white>.look_up(from, bl) << std::endl;
    std::cout << chess::pawn_attack_tbl<chess::color::white>.look_up(from) << std::endl;
  }

  {
    std::cout << "test4:" << std::endl;
    const auto from = chess::square::from_index(chess::tbl_square({1, 1}).index());
    chess::square_set bl{};
    std::cout << bl << std::endl;
    std::cout << from << std::endl;
    std::cout << chess::pawn_push_tbl<chess::color::white>.look_up(from, bl) << std::endl;
    std::cout << chess::pawn_attack_tbl<chess::color::white>.look_up(from) << std::endl;
  }

  {
    std::cout << "test5:" << std::endl;
    const auto from = chess::square::from_index(chess::tbl_square({1, 6}).index());
    chess::square_set bl{};
    std::cout << bl << std::endl;
    std::cout << from << std::endl;
    std::cout << chess::pawn_push_tbl<chess::color::black>.look_up(from, bl) << std::endl;
    std::cout << chess::pawn_attack_tbl<chess::color::black>.look_up(from) << std::endl;
  }
  
  {
    std::cout << "test5:" << std::endl;
    const auto from = chess::square::from_index(chess::tbl_square({1, 4}).index());
    chess::square_set bl{};
    std::cout << bl << std::endl;
    std::cout << from << std::endl;
    std::cout << chess::pawn_push_tbl<chess::color::black>.look_up(from, bl) << std::endl;
    std::cout << chess::pawn_attack_tbl<chess::color::black>.look_up(from) << std::endl;
  }

}
