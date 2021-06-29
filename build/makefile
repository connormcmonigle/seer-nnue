EXE = seer
SRC = ../src/seer.cc
CC = g++

EVALFILE = weights/0xbd25a7d4.bin

OPSLIMIT = 1000000000
CXXSTANDARD = 17
INCLUDE = ../include

CXXFLAGS += -std=c++$(CXXSTANDARD)
CXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
CXXFLAGS += -Wall -Wextra
CXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
CXXFLAGS += -I$(INCLUDE)
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