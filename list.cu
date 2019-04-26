#include "list.h"
#include <assert.h>
#include <stdio.h>

list *list_create(int capacity) {
	list list_cpu;
	list *list_gpu;
	list_cpu.length = 0;
	list_cpu.capacity = capacity;
	HANDLE_RESULT(cudaMalloc(&(list_cpu.arr), (capacity + 1) * sizeof(state*)));
	HANDLE_RESULT(cudaMalloc(&list_gpu, sizeof(struct list)));
	HANDLE_RESULT(cudaMemcpy(list_gpu, &list_cpu, sizeof(struct list),
				cudaMemcpyDefault));
	return list_gpu;
}

void list_destroy(list *list_gpu) {
	list list_cpu;
	HANDLE_RESULT(cudaMemcpy(&list_cpu, list_gpu, sizeof(struct list),
				cudaMemcpyDefault));
	HANDLE_RESULT(cudaFree(list_cpu.arr));
	HANDLE_RESULT(cudaFree(list_gpu));
}
__device__ void list_clear(list *list) {
	list->length = 0;
}

__device__ void list_insert(list *list, state *state) {
	assert(list->length < list->capacity);
	list->arr[list->length++] = state;
}

__device__ void list_remove(list *list, int index) {
	assert(list->length < list->capacity);
	list->arr[index] = NULL;
}

__device__ state *list_get(list *list, int index) {
	assert(index < list->length);
	return list->arr[index];
}

