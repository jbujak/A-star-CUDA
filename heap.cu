#include "heap.h"
#include "astar_gpu.h"
#include <stdlib.h>
#include <stdio.h>

static void swap(state **s1, state **s2);

heap *heap_create(int capacity) {
	heap *heap = (struct heap*)malloc(sizeof(struct heap));
	heap->size = 0;
	heap->states = (state**)calloc(capacity + 1, sizeof(state*));
	return heap;
}

int a = 0;
void heap_insert(heap *heap, state *state) {
	heap->size++;
	heap->states[heap->size] = state;
	int current = heap->size;
	while (current > 1 && heap->states[current]->f < heap->states[current / 2]->f) {
		swap(&(heap->states[current]), &(heap->states[current / 2]));
		current /= 2;
	}
}

state *heap_extract(heap *heap) {
	state *res = heap->states[1];
	heap->states[1] = heap->states[heap->size];
	heap->states[heap->size] = NULL;
	heap->size--;
	int current = 1;
	while (current < heap->size) {
		int smallest = current;
		int child = 2 * current;
		if (child <= heap->size && heap->states[child]->f < heap->states[smallest]->f) {
			smallest = child;
		}
		child = 2 * current + 1;
		if (child <= heap->size && heap->states[child]->f < heap->states[smallest]->f) {
			smallest = child;
		}
		if (smallest == current) {
			break;
		}
		swap(&(heap->states[current]), &(heap->states[smallest]));
		current = smallest;
	}
	return res;
}

void heap_destroy(heap *heap) {
	free(heap->states);
	free(heap);
}


static void swap(state **s1,  state **s2) {
	state *tmp = *s1;
	*s1 = *s2;
	*s2 = tmp;
}
