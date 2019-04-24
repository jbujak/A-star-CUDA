#include <stdio.h>

#include "astar_gpu.h"
#include "heap.h"

void print_expanded(char **expanded) {
	for (int i = 0; expanded[i] != NULL; i++) {
		printf("%s\n", expanded[i]);
	}
}

int astar_cpu(const char *s, const char *t, int k, expand_fun expand) {
	heap *heap = heap_create(10);
	state states[10];
	for (int i = 0; i < 10; i++) {
		states[i].f = (i + 5) % 10;
	}
	for (int i = 0; i < 10; i++) {
		heap_insert(heap, &(states[i]));
	}
	while (heap->size > 0) {
		printf("%d\n", heap_extract(heap)->f);
	}
	heap_destroy(heap);
	return 0;
}
