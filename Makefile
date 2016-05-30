SRC=convolution.c ppmparser.c
LIB=lib
DST=convolution-hyb
OPTS=-std=c99 -Wall -Wextra -pedantic-errors -O2 -g
INC=-fopenmp
CC=mpicc

conv-hyb: $(SRC) $(LIB)
	@$(CC) $(OPTS) $(SRC) -L $(LIB) -o $(DST) $(INC)
	@echo Compilation complete!

clean: $(DST)
	@$(RM) $(DST)

