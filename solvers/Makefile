TARGETS := $(patsubst %.c,%.ll,$(wildcard *.c))

all: ${TARGETS}

%.ll: %.c solvers.h
	clang -O3 -S -emit-llvm -o $@ $<
