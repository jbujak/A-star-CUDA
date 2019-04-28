FLAGS = -std=c++14 -g -G
OBJS = astar_gpu.o heap.o list.o sliding_puzzle.o cuda_utils.o pathfinding.o
ifeq ($(wildcard /opt/cuda/bin/nvcc),)
	NVCC=nvcc
else
	NVCC=/opt/cuda/bin/nvcc
endif

astar_gpu: main.cu $(OBJS)
	$(NVCC) $(FLAGS) main.cu $(OBJS) -o astar_gpu

%.o: %.cu %.h
	$(NVCC) $(FLAGS) -c --device-c $*.cu -o $*.o

.PHONY: clean
clean:
	rm *.o
	rm astar_gpu
