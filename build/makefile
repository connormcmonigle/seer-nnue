EXE = seer
CXX = g++

CXXSTANDARD = 17
EVALFILE = weights/q0xb6df28fb.bin
OPSLIMIT = 1000000000

CXXSRC += $(wildcard ../src/*.cc )
CXXSRC += $(wildcard ../src/search/*.cc)
CXXSRC += $(wildcard ../src/chess/*.cc)
CXXSRC += $(wildcard ../src/engine/*.cc)
CXXSRC += ../syzygy/tbprobe.cc

INCLUDE = ../include
INCBIN = ../incbin
SYZYGY = ../syzygy

BASECXXFLAGS += -std=c++$(CXXSTANDARD)
BASECXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
BASECXXFLAGS += -Wall -Wextra -pedantic
BASECXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
BASECXXFLAGS += -DEVALFILE=\"$(EVALFILE)\"
CXXFLAGS += $(BASECXXFLAGS)


CPPFLAGS += -MMD
CPPFLAGS += -I$(INCLUDE)
CPPFLAGS += -I$(INCBIN)
CPPFLAGS += -I$(SYZYGY)

LDFLAGS = -lpthread

CXXOBJECTS += $(CXXSRC:%.cc=%.o)
CXXDEPENDS += $(CXXSRC:%.cc=%.d)

.PHONY: default
default: flto

.PHONY: pgo
pgo:
	$(MAKE) clean
	$(MAKE) profileclean
	$(MAKE) profilegenerate EXE=$(EXE) CXX=$(CXX) EVALFILE=$(EVALFILE)
	./$(EXE) bench
	$(MAKE) clean
	$(MAKE) profileuse EXE=$(EXE) CXX=$(CXX) EVALFILE=$(EVALFILE)
	$(MAKE) profileclean

.PHONY: profilegenerate
profilegenerate: CXXFLAGS += -fprofile-generate
profilegenerate: flto

.PHONY: profileuse
profileuse: CXXFLAGS += -fprofile-use
profileuse: flto

.PHONY: flto
flto: CXXFLAGS += -flto -fwhole-program
flto: binary

.PHONY: plain
flto: binary

.PHONY: binary
binary: $(CXXOBJECTS)
	+$(CXX) $(CXXFLAGS) -o $(EXE) $^ $(LDFLAGS)

.PHONY: clean
clean:
	rm -f $(CXXOBJECTS) $(CXXDEPENDS)

.PHONY: profileclean
profileclean:
	rm -f $(CXXSRC:%.cc=%.gcda)


-include $(CXXDEPENDS)
