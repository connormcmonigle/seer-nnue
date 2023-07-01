EXE = seer
SRC = ../src/seer.cc ../syzygy/tbprobe.c ../src/search/*.cc ../src/chess/*.cc ../src/engine/*.cc
CC = g++

EVALFILE = weights/0xddf7a131.bin

OPSLIMIT = 1000000000
CXXSTANDARD = 17

INCLUDE = ../include
INCBIN = ../incbin
SYZYGY = ../syzygy

CXXFLAGS += -std=c++$(CXXSTANDARD)
CXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp -flto -fwhole-program
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
CXXFLAGS += -I$(INCLUDE) -I$(INCBIN) -I$(SYZYGY)
CXXFLAGS += -DEVALFILE=\"$(EVALFILE)\"

THREADSANFLAGS += $(CXXFLAGS) -fsanitize=thread
ADDRESSSANFLAGS += $(CXXFLAGS) -fsanitize=address
UBSANFLAGS += $(CXXFLAGS) -fsanitize=undefined
TUNEFLAGS += $(CXXFLAGS) -DTUNE


LIBS = -lpthread

default:
	$(CC) $(SRC) $(CXXFLAGS) $(LIBS) -o $(EXE)

pgo:
	rm -f *.gcda
	$(CC) -fprofile-generate $(SRC) $(CXXFLAGS) $(LIBS) -o $(EXE)
	./$(EXE) bench
	$(CC) -fprofile-use $(SRC) $(CXXFLAGS) $(LIBS) -o $(EXE)
	rm -f *.gcda

threadsan:
	$(CC) $(SRC) $(THREADSANFLAGS) $(LIBS) -o $(EXE)

addresssan:
	$(CC) $(SRC) $(ADDRESSSANFLAGS) $(LIBS) -o $(EXE)

ubsan:
	$(CC) $(SRC) $(UBSANFLAGS) $(LIBS) -o $(EXE)

tune:
	$(CC) $(SRC) $(TUNEFLAGS) $(LIBS) -o $(EXE)
