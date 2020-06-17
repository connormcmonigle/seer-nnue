cmake_minimum_required(VERSION 3.10)
project(nnue)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-O3 -march=native -fopenmp -fconstexpr-steps=100000000 -Wall -Wextra")

include_directories (include)

add_executable (test_propagate test/test_propagate.cc)
add_executable (test_tbl test/test_tbl.cc)
add_executable (test_crtp test/test_crtp.cc)
add_executable (test_board test/test_board.cc)
add_executable (test_perft test/perft.cc)
