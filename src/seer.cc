/*
  Seer is a UCI chess engine by Connor McMonigle
  Copyright (C) 2021  Connor McMonigle

  Seer is free software: you can redistribute it and/or modify
  it under the terms of the GNU General Public License as published by
  the Free Software Foundation, either version 3 of the License, or
  (at your option) any later version.

  Seer is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.
  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#include <uci.h>

#include <chrono>
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
  engine::uci uci{};

  const bool perform_bench = (argc == 2) && (std::string(argv[1]) == "bench");
  if (perform_bench) {
    uci.bench();
    return 0;
  }

  for (std::string line{}; !uci.should_quit() && std::getline(std::cin, line);) { uci.read(line); }
}
