#ifndef HEAP_H
#define HEAP_H

#include "astar_gpu.h"

struct heap {
	state **states;
	int size;
};

heap *heap_create(int capacity);

void heap_destroy(heap *heap_dev);

__device__ void heap_insert(heap *heap, state *state);

__device__ state *heap_extract(heap *heap);

#endif //HEAP_H
