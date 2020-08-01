cmake_minimum_required(VERSION 3.10)
project(nnue)

set (CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "-O3 -fconstexpr-ops-limit=1000000000  -march=native -fopenmp -Wall -Wextra")

include_directories (include)

add_executable (test_propagate test/test_propagate.cc)
add_executable (test_tbl test/test_tbl.cc)
add_executable (test_crtp test/test_crtp.cc)
add_executable (test_board test/test_board.cc)
add_executable (test_perft test/perft.cc)
add_executable (test_hash test/test_hash.cc)
add_executable (test_parse_fen test/test_parse_fen.cc)
add_executable (test_half_kp_indices test/test_half_kp_indices.cc)
add_executable (test_after_uci_moves test/test_after_uci_moves.cc)
add_executable (test_nnue_perf test/test_nnue_perf.cc)
add_executable (test_move_orderer test/test_move_orderer.cc)
add_executable (test_option_parser test/test_option_parser.cc)

add_executable (test_history_heuristic test/test_history_heuristic.cc)
target_link_libraries (test_history_heuristic pthread)


add_executable (test_worker test/test_worker.cc)
target_link_libraries (test_worker pthread)

add_executable (test_tt test/test_tt.cc)
target_link_libraries (test_tt pthread)

add_executable (seer src/seer.cc)
target_link_libraries (seer pthread)
