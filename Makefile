FLAGS=-std=c++14
ifeq ($(wildcard /opt/cuda/bin/nvcc),)
	NVCC=nvcc
else
	NVCC=/opt/cuda/bin/nvcc
endif

astar_gpu: main.cpp
	$(NVCC) $(FLAGS) main.cpp -o astar_gpu
