#include <iostream>
#include <thread>

#include <board.h>
#include <transposition_table.h>

int main(){
  constexpr size_t tbl_size = 1024;
  const auto mv_ls = chess::board::start_pos().generate_moves();
  const auto mv = mv_ls.data[8];
  std::cout << "mv: " << mv << std::endl;
  {
    std::cout << "test nominal multithread:\n";
    chess::table tt(tbl_size);
    const auto hash_0 = zobrist::random_bit_string();
    const auto hash_1 = zobrist::random_bit_string();
    const auto entry_0 = chess::tt_entry(hash_0, chess::bound_type::lower, 0.1, mv, 10);
    const auto entry_1 = chess::tt_entry(hash_1, chess::bound_type::upper, 0.3, mv, 5);

    std::thread t0([&tt, entry_0](){ tt.insert(entry_0); });
    std::thread t1([&tt, entry_1](){ tt.insert(entry_1); });

    t0.join();
    t1.join();

    if(auto res = tt.find(entry_0.key()); res != tt.end()){
      std::cout << *res << std::endl;
    }
    if(auto res = tt.find(entry_1.key()); res != tt.end()){
      std::cout << *res << std::endl;
    }
  }

  {
    std::cout << "test collision single-thread:\n";
    chess::table tt(tbl_size);
    const auto hash = zobrist::random_bit_string();
    const auto entry_0 = chess::tt_entry(hash, chess::bound_type::lower, 0.1, mv, 5);
    const auto entry_1 = chess::tt_entry(hash, chess::bound_type::upper, 0.9, mv, 6);

    tt.insert(entry_0);
    tt.insert(entry_1);

    if(auto res = tt.find(hash); res != tt.end()){
      std::cout << *res << std::endl;
    }
  }

  {
    std::cout << "test collision multithread:\n";
    chess::table tt(tbl_size);
    const auto hash = zobrist::random_bit_string();
    const auto entry_0 = chess::tt_entry(hash, chess::bound_type::upper, 0.4, mv, 7);
    const auto entry_1 = chess::tt_entry(hash, chess::bound_type::lower, 0.9, mv, 8);

    std::thread t0([&tt, entry_1](){ tt.insert(entry_1); });
    std::thread t1([&tt, entry_0](){ tt.insert(entry_0); });

    t0.join();
    t1.join();

    if(auto res = tt.find(hash); res != tt.end()){
      std::cout << *res << std::endl;
    }else{
      //This seems to never happen in practice
      std::cout << "data was mutilated by collision and multiple threads writing simultaneously?\n";
    }
  }
}


