#ifndef HEAP_H
#define HEAP_H

#include "astar_gpu.h"

struct heap {
	state **states;
	int size;
};

heap *heap_create(int capacity);

void heap_insert(heap *heap, state *state);

state *heap_extract(heap *heap);

void heap_destroy(heap *heap);

#endif //HEAP_H
