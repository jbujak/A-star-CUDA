#include <stdio.h>

#include "astar_gpu.h"
#include "heap.h"
#include "list.h"
#include "sliding_puzzle.h"
#include "cuda_utils.h"

#define STATES (1024 * 1024)
#define HASH_SIZE  (1024 * 1024)
#define HASH_FUNS 128

__global__ void astar_kernel(const char *s, const char *t, int k,
		heap **Q, list *S, state **H, state *states_pool, char **expand_buf,
		expand_fun expand, heur_fun h);
__device__ void hash_with_replacement_deduplicate(state **H, list *T);
__device__ int f(const state *x, const char *t, heur_fun h);


char **expand_buf_create(int elements, int element_size);

state *states_pool_create();
void states_pool_destroy(state *states_pull);
__device__ state *state_create(const char *node, int f, int g, state *prev,
		state *states_pool);

__device__ void print_expanded(char **expanded) {
	for (int i = 0; expanded[i] != NULL; i++) {
		printf("%s\n", expanded[i]);
	}
}


int astar_gpu(const char *s_in, const char *t_in, int k) {
	char *s_gpu, *t_gpu;
	expand_fun expand_fun_cpu;
	heur_fun h_cpu;
	int expand_elements;
	int expand_element_size;

	sliding_puzzle_preprocessing(s_in, t_in, &s_gpu, &t_gpu, &expand_fun_cpu, &h_cpu,
			&expand_elements, &expand_element_size);

	state **H;
	char **expand_buf = expand_buf_create(expand_elements, expand_element_size);
	HANDLE_RESULT(cudaMalloc(&H, HASH_SIZE * sizeof(state*)));
	heap **Q = heaps_create(k);
	list *S = list_create(100);
	state *states_pool = states_pool_create();

	astar_kernel<<<1, 1>>>(s_gpu, t_gpu, k, Q, S, H, states_pool, expand_buf,
			expand_fun_cpu, h_cpu);

	states_pool_destroy(states_pool);
	list_destroy(S);
	heaps_destroy(Q, k);
	HANDLE_RESULT(cudaFree(H));
	HANDLE_RESULT(cudaDeviceSynchronize());
	return 0;
}

__global__ void astar_kernel(const char *s, const char *t, int k,
		heap **Q, list *S, state **H, state *states_pool, char **expand_buf,
		expand_fun expand, heur_fun h) {
	state *m = NULL;

	heap_insert(Q[0], state_create(s, 0, 0, NULL, states_pool));

	int steps = 0;
	while (!heaps_empty(Q, k)) {
		steps++;
		list_clear(S);
		for (int i = 0; i < k; i++) {
			if (Q[i]->size == 0) continue;
			state *q = heap_extract(Q[i]);
			//printf("%s\n", q->node);
			if (steps % 1000 == 0) printf("distance: %d\n", h(q->node, t));
			if (cuda_str_eq(q->node, t)) {
				if (m == NULL || f(q, t, h) < f(m, t, h)) {
					m = q;
				}
				continue;
			}
			expand(q->node, expand_buf);
			for (int j = 0; expand_buf[j] != NULL; j++) {
				list_insert(S, state_create(expand_buf[j], -1, q->g + 1, q, states_pool));
			}
		}
		if (m != NULL && f(m, t, h) < heaps_min(Q, k)) {
			printf("In %d steps: Found path of length %d: [\n", steps, m->g);
			state *cur = m;
			while (cur != NULL) {
				for (int i = 0; i < 25; i++) {
					printf("%c%c ", cur->node[3 * i], cur->node[3 * i + 1]);
					if (i % 5 == 4) printf("\n");
				}
				printf("\n");
				cur = cur->prev;
			}
			printf("]\n");
			break;
		}
		hash_with_replacement_deduplicate(H, S);
		for (int i = 0; i < S->length; i++) {
			state *t1 = list_get(S, i);
			if (t1 != NULL) {
				t1->f = f(t1, t, h);
				heap_insert(Q[0], t1);
			}
		}
	}
}

__device__ void hash_with_replacement_deduplicate(state **H, list *T) {
	for (int i = 0; i < T->length; i++) {
		int z = 0;
		state *t = list_get(T, i);
		for (int j = 0; j < HASH_FUNS; j++) {
			state *el = H[jenkins_hash(j, t->node) % HASH_SIZE];
			if (el == NULL || cuda_str_eq(t->node, el->node)) {
				z = j;
				break;
			}
		}
		int index = jenkins_hash(z, t->node) % HASH_SIZE;
		t = (state*)atomicExch((unsigned long long*)&(H[index]), (unsigned long long)t);
		if (t != NULL && cuda_str_eq(t->node, list_get(T, i)->node)) {
			list_remove(T, i);
			continue;
		}
		t = list_get(T, i);
		for (int j = 0; j < HASH_FUNS; j++) {
			if (j != z) {
				state *el = H[jenkins_hash(j, t->node) % HASH_SIZE];
				if (el != NULL && cuda_str_eq(el->node, t->node)) {
					list_remove(T, i);
					break;
				}
			}
		}
	}
}

__device__ int f(const state *x, const char *t, heur_fun h) {
	return x->g + h(x->node, t);
}

state *states_pool_create() {
	state *states_pool = NULL;
	HANDLE_RESULT(cudaMalloc(&states_pool, STATES * sizeof(state)));
	HANDLE_RESULT(cudaMemset(states_pool, 0, STATES * sizeof(state)));
	return states_pool;
}

void states_pool_destroy(state *states_pull) {
	HANDLE_RESULT(cudaFree(states_pull));
}

char **expand_buf_create(int elements, int element_size) {
	elements++; // For terminating NULL element
	element_size++; // For terminating NULL char
	char **buf_cpu = (char**)malloc(elements * sizeof(char*));
	for (int i = 0; i < elements; i++) {
		HANDLE_RESULT(cudaMalloc(&(buf_cpu[i]), element_size));
	}
	char **buf_gpu;
	HANDLE_RESULT(cudaMalloc(&buf_gpu, elements * sizeof(char*)));
	HANDLE_RESULT(cudaMemcpy(buf_gpu, buf_cpu, elements * sizeof(char),
				cudaMemcpyDefault));
	HANDLE_RESULT(cudaDeviceSynchronize());
	free(buf_cpu);
	return buf_gpu;

}

__device__ int used_states = 0;
__device__ state *state_create(const char *node, int f, int g, state *prev,
		state *states_pool) {
	int index = atomicAdd(&used_states, 1);
	state *result = &(states_pool[index]);
	result->node = node;
	result->f = f;
	result->g = g;
	result->prev = prev;
	return result;
}

