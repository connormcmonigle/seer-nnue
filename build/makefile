EXE = seer
SRC = ../
CXX = g++

WEIGHTSFILE = weights/0x319651dd.bin

OPSLIMIT = 1000000000
CXXSTANDARD = 17
INCLUDE = ../include

CXXFLAGS += -std=c++$(CXXSTANDARD)
CXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
CXXFLAGS += -I$(INCLUDE)
CXXFLAGS += -DWEIGHTSFILE=\"$(WEIGHTSFILE)\"

LIBS = -lpthread

SRC = ../src/seer.cc

default:
	$(CXX) $(SRC) $(CXXFLAGS) $(LIBS) -o $(EXE)