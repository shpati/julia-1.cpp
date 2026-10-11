CC      ?= cc
# -march=native gives the biggest speedup (AVX2/AVX512 FMA); drop it for a portable binary.
CFLAGS  ?= -O3 -march=native -std=c11 -Wall -Wextra
LDLIBS  += -lm
# Multithreading: `make OMP=1` (needs an OpenMP-capable compiler).
ifeq ($(OMP),1)
CFLAGS  += -fopenmp
LDFLAGS += -fopenmp
endif

SRC = main.c julia.c tokenizer.c json.c

julia-1: $(SRC) julia.h tokenizer.h json.h
	$(CC) $(CFLAGS) $(LDFLAGS) -o $@ $(SRC) $(LDLIBS)

clean:
	rm -f julia-1

.PHONY: clean
