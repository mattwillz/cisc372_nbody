FLAGS= -DDEBUG
LIBS= -lm
ALWAYS_REBUILD=makefile

CC = gcc
NVCC = nvcc

nbody: nbody.o compute.o
	$(NVCC) $(FLAGS) $^ -o $@ $(LIBS)

nbody.o: nbody.c planets.h config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(CC) $(FLAGS) -c $<

compute.o: compute.c config.h vector.h compute.h $(ALWAYS_REBUILD)
	$(NVCC) $(FLAGS) -x cu -c $<

clean:
	rm -f *.o nbody 
