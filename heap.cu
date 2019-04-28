#include "heap.h"
#include "astar_gpu.h"
#include "cuda_utils.h"
#include <stdlib.h>
#include <stdio.h>

__device__ static void swap(state **s1, state **s2);

heap **heaps_create(int k) {
	heap **Q_cpu = (heap**)malloc(k * sizeof(heap*));
	heap **Q_dev = NULL;
	for (int i = 0; i < k; i++) {
		Q_cpu[i] = heap_create(16 * 1024);
	}
	HANDLE_RESULT(cudaMalloc(&Q_dev, k * sizeof(heap*)));
	HANDLE_RESULT(cudaMemcpy(Q_dev, Q_cpu, k * sizeof(heap*), cudaMemcpyDefault));
	free(Q_cpu);
	return Q_dev;
}

heap *heap_create(int capacity) {
	heap heap_cpu;
	heap *heap_dev;
	heap_cpu.size = 0;
	HANDLE_RESULT(cudaMalloc(&(heap_cpu.states), (capacity + 1) * sizeof(state*)));
	HANDLE_RESULT(cudaMemset(heap_cpu.states, 0, (capacity + 1) * sizeof(state*)));

	HANDLE_RESULT(cudaMalloc(&heap_dev, sizeof(heap)));
	HANDLE_RESULT(cudaMemcpy(heap_dev, &heap_cpu, sizeof(heap), cudaMemcpyDefault));
	return heap_dev;
}

void heaps_destroy(heap **Q_dev, int k) {
	heap **Q_cpu = (heap**)malloc(k * sizeof(heap*));
	HANDLE_RESULT(cudaMemcpy(Q_cpu, Q_dev, k * sizeof(heap*), cudaMemcpyDefault));
	for (int i = 0; i < k; i++) {
		heap_destroy(Q_cpu[i]);
	}
	free(Q_cpu);
	HANDLE_RESULT(cudaFree(Q_dev));
}

void heap_destroy(heap *heap_dev) {
	heap heap_cpu;
	HANDLE_RESULT(cudaMemcpy(&heap_cpu, heap_dev, sizeof(heap), cudaMemcpyDefault));
	HANDLE_RESULT(cudaFree(heap_cpu.states));
	HANDLE_RESULT(cudaFree(heap_dev));
}

__device__ void heap_insert(heap *heap, state *state) {
	heap->size++;
	heap->states[heap->size] = state;
	int current = heap->size;
	while (current > 1 && heap->states[current]->f < heap->states[current / 2]->f) {
		swap(&(heap->states[current]), &(heap->states[current / 2]));
		current /= 2;
	}
}

__device__ state *heap_extract(heap *heap) {
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

__device__ bool heaps_empty(heap **heaps, int k) {
	for (int i = 0; i < k; i++) {
		if (heaps[i]->size != 0) return false;
	}
	return true;
}

__device__ int heaps_min(heap **heaps, int k) {
	int best_f = INT_MAX;
	for (int i = 0; i < k; i++) {
		state *current_best = heaps[i]->states[1];
		if (current_best != NULL && current_best->f < best_f) {
			best_f = current_best->f;
		}
	}
	return best_f;
}


__device__ static void swap(state **s1,  state **s2) {
	state *tmp = *s1;
	*s1 = *s2;
	*s2 = tmp;
}
