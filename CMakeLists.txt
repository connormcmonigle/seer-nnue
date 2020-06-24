cmake_minimum_required(VERSION 3.10)
project(nnue)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -Wall -Wextra")

include_directories (include)

add_executable (test_propagate test/test_propagate.cc)
add_executable (test_tbl test/test_tbl.cc)
add_executable (test_crtp test/test_crtp.cc)
add_executable (test_board test/test_board.cc)
add_executable (test_perft test/perft.cc)
add_executable (test_hash test/test_hash.cc)
add_executable (test_parse_fen test/test_parse_fen.cc)
add_executable (test_half_kp_indices test/test_half_kp_indices.cc)
add_executable (test_pv_search test/test_pv_search.cc)
add_executable (test_after_uci_moves test/test_after_uci_moves.cc)

add_executable (test_worker test/test_worker.cc)
target_link_libraries (test_worker pthread)

add_executable (test_tt test/test_tt.cc)
target_link_libraries (test_tt pthread)
