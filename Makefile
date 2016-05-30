SRC=convolution.c ppmparser.c
LIB=lib
DST=convolution-mpi
OPTS=-std=c99 -Wall -Wextra -pedantic-errors -O2 -g
CC=mpicc

conv-omp: $(SRC) $(LIB)
	@$(CC) $(OPTS) $(SRC) -L $(LIB) -o $(DST)
	@echo Compilation complete!

clean: $(DST)
	@$(RM) $(DST)

