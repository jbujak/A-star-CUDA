FLAGS=-std=c++14
NVCC=/opt/cuda/bin/nvcc
#NVCC=nvcc
astar_gpu: main.cpp
	$(NVCC) $(FLAGS) main.cpp -o astar_gpu
