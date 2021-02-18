cmake_minimum_required (VERSION 3.10)
project (nnue)

find_package (Threads REQUIRED)

set (CMAKE_CXX_STANDARD 17)

set (OPS_LIMIT 1000000000)

if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set (CMAKE_CXX_FLAGS "-Ofast -g -DNDEBUG -fconstexpr-ops-limit=${OPS_LIMIT} -march=native -mtune=native -fopenmp -Wall -Wextra")
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set (CMAKE_CXX_FLAGS "-Ofast -g -DNDEBUG -fconstexpr-steps=${OPS_LIMIT} -march=native -mtune=native -fopenmp -Wall -Wextra")
endif()

include_directories (include)

add_executable (test_propagate test/test_propagate.cc)
add_executable (test_tbl test/test_tbl.cc)
add_executable (test_crtp test/test_crtp.cc)
add_executable (test_board test/test_board.cc)
add_executable (test_perft test/perft.cc)
add_executable (test_hash test/test_hash.cc)
add_executable (test_parse_fen test/test_parse_fen.cc)
add_executable (test_feature_indices test/test_feature_indices.cc)
add_executable (test_after_uci_moves test/test_after_uci_moves.cc)
add_executable (test_move_orderer test/test_move_orderer.cc)
add_executable (test_option_parser test/test_option_parser.cc)
add_executable (test_lva test/test_lva.cc)
add_executable (test_see test/test_see.cc)
add_executable (test_bit_field test/test_bit_field.cc)
add_executable (test_is_passed test/test_is_passed.cc)
add_executable (test_history test/test_history.cc)
add_executable (test_search_stack test/test_search_stack.cc)
add_executable (test_history_heuristic test/test_history_heuristic.cc)
add_executable (test_time_manager test/test_time_manager.cc)
add_executable (test_streamer test/test_streamer.cc)
add_executable (test_training test/test_training.cc)

add_executable (test_worker test/test_worker.cc)
target_link_libraries (test_worker Threads::Threads)

add_executable (test_tt test/test_tt.cc)
target_link_libraries (test_tt Threads::Threads)

add_executable (seer src/seer.cc)
target_link_libraries (seer Threads::Threads)