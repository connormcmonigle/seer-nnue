#include <iostream>
#include <set>

#include <enum_util.h>
#include <board.h>



struct tester : chess::sided<tester, std::set<size_t>> {
  std::set<size_t> white{};
  std::set<size_t> black{};
  tester(){}
};

bool operator!=(const tester& a, const tester& b){
  return a.white != b.white || a.black != b.black;
}

std::ostream& operator<<(std::ostream& ostr, const tester& test){
  ostr << "tester(white={";
  for(auto elem : test.white){ ostr << elem << ", "; }
  ostr << "}, black={";
  for(auto elem : test.black){ ostr << elem << ", "; }
  return ostr << "})";
}

int main(){

  auto gen = std::mt19937(std::random_device()());
  std::cout << "enter number of test games to run: ";
  size_t game_count; std::cin >> game_count;

  for(size_t n(0); n < game_count; ++n){
    auto bd = chess::board::start_pos();
    tester updatable{};
    bd.show_init(updatable);

    for(;;){
      std::cout << bd.fen() << std::endl;
      std::cout << updatable << std::endl;
      const chess::move_list mv_ls = bd.generate_moves();
    
      if(mv_ls.size() == 0 || bd.lat_.move_count > 500){
        std::cout << "SUCCESS\n";
        break;
      }

      tester foil{};
      bd.show_init(foil);

      if(foil != updatable){
        std::cout << "FAIL\n";
        std::terminate();
      }

      std::uniform_int_distribution<size_t> dist(0, mv_ls.size() - 1);
      const chess::move mv = mv_ls.data[dist(gen)];
      updatable = bd.half_kp_updated(mv, updatable);
      bd = bd.forward(mv);
    }
  }
}
