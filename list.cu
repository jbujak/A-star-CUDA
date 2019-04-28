#include "list.h"
#include "cuda_utils.h"
#include <assert.h>
#include <stdio.h>

list **lists_create(int lists, int capacity) {
	list **lists_cpu = (list**)malloc(lists * sizeof(list*));
	list **lists_gpu = NULL;
	for (int i = 0; i < lists; i++) {
		lists_cpu[i] = list_create(capacity);
	}
	HANDLE_RESULT(cudaMalloc(&lists_gpu, lists * sizeof(list*)));
	HANDLE_RESULT(cudaMemcpy(lists_gpu, lists_cpu, lists * sizeof(list*), cudaMemcpyDefault));
	free(lists_cpu);
	return lists_gpu;
}

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

void lists_destroy(list **lists_gpu, int lists) {
	list **lists_cpu = (list**)malloc(lists * sizeof(list*));
	HANDLE_RESULT(cudaMemcpy(lists_cpu, lists_gpu, lists * sizeof(list*), cudaMemcpyDefault));
	for (int i = 0; i < lists; i++) {
		list_destroy(lists_cpu[i]);
	}
	HANDLE_RESULT(cudaFree(lists_gpu));
	free(lists_cpu);
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
	int index = atomicAdd(&(list->length), 1);
	assert(index < list->capacity);
	list->arr[index] = state;
}

__device__ void list_remove(list *list, int index) {
	assert(list->length < list->capacity);
	list->arr[index] = NULL;
}

__device__ state *list_get(list *list, int index) {
	assert(index < list->length);
	return list->arr[index];
}

