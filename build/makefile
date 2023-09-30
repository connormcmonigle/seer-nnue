EXE = seer
CXX = g++

CXXSTANDARD = 17
EVALFILE = weights/0x2b629831.bin
OPSLIMIT = 1000000000

CXXSRC += $(wildcard ../src/*.cc )
CXXSRC += $(wildcard ../src/search/*.cc)
CXXSRC += $(wildcard ../src/chess/*.cc)
CXXSRC += $(wildcard ../src/engine/*.cc)
CXXSRC += ../syzygy/tbprobe.cc

INCLUDE = ../include
INCBIN = ../incbin
SYZYGY = ../syzygy

CXXFLAGS += -std=c++$(CXXSTANDARD)
CXXFLAGS += -O3 -g -DNDEBUG -march=native -mtune=native -fopenmp
CXXFLAGS += -Wall -Wextra -pedantic
CXXFLAGS += -fconstexpr-ops-limit=$(OPSLIMIT)
CXXFLAGS += -DEVALFILE=\"$(EVALFILE)\"

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
