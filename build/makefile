EXE = seer

CC = gcc
CXX = g++

CXXSTANDARD = 17
CSTANDARD = 17

EVALFILE = weights/0xddf7a131.bin
OPSLIMIT = 1000000000

CXXSRC += $(wildcard ../src/*.cc )
CXXSRC += $(wildcard ../src/search/*.cc)
CXXSRC += $(wildcard ../src/chess/*.cc)
CXXSRC += $(wildcard ../src/engine/*.cc)

CSRC := ../syzygy/tbprobe.c

INCLUDE = ../include
INCBIN = ../incbin
SYZYGY = ../syzygy

CXXFLAGS += -std=c++$(CXXSTANDARD)
CXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
CXXFLAGS += -Wall -Wextra -pedantic
CXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
CXXFLAGS += -DEVALFILE=\"$(EVALFILE)\"

CFLAGS += -std=c$(CSTANDARD)
CFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
CFLAGS += -Wall -Wextra -pedantic

CPPFLAGS += -MMD
CPPFLAGS += -I$(INCLUDE)
CPPFLAGS += -I$(INCBIN)
CPPFLAGS += -I$(SYZYGY)

LDFLAGS = -lpthread

COBJECTS += $(CSRC:%.c=%.o)
CDEPENDS += $(CSRC:%.c=%.d)

CXXOBJECTS += $(CXXSRC:%.cc=%.o)
CXXDEPENDS += $(CXXSRC:%.cc=%.d)

.PHONY: default
default: flto

.PHONY: pgo
pgo:
	$(MAKE) clean
	$(MAKE) profileclean
	$(MAKE) profilegenerate EXE=$(EXE) CC=$(CC) CXX=$(CXX) EVALFILE=$(EVALFILE)
	./$(EXE) bench
	$(MAKE) clean
	$(MAKE) profileuse EXE=$(EXE) CC=$(CC) CXX=$(CXX) EVALFILE=$(EVALFILE)
	$(MAKE) profileclean

.PHONY: profilegenerate
profilegenerate: CXXFLAGS += -fprofile-generate
profilegenerate: CFLAGS += -fprofile-generate
profilegenerate: flto

.PHONY: profileuse
profileuse: CXXFLAGS += -fprofile-use
profileuse: CFLAGS += -fprofile-use
profileuse: flto

.PHONY: flto
flto: CXXFLAGS += -flto -fwhole-program
flto: CFLAGS += -flto -fwhole-program
flto: binary

.PHONY: plain
plain: binary

.PHONY: binary
binary: $(CXXOBJECTS) $(COBJECTS)
	+$(CXX) $(CXXFLAGS) -o $(EXE) $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(CXXOBJECTS) $(CXXDEPENDS)
	rm -f $(COBJECTS) $(CDEPENDS)

.PHONY: profileclean
profileclean:
	rm -f $(CSRC:%.c=%.gcda)
	rm -f $(CXXSRC:%.cc=%.gcda)


-include $(CXXDEPENDS) $(CDEPENDS)
