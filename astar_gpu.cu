#include <stdio.h>

#include "astar_gpu.h"
#include "heap.h"
#include "list.h"

__global__ void astar_kernel(const char *s, const char *t, int k,
		heap **Q, list *S, state **H, state *states_pool, char **expand_buf,
		expand_fun expand, heur_fun h);

__device__ int f(const state *x, const char *t, heur_fun h);

__device__ unsigned int jenkins_hash(int j, const char *str);
__device__ void hash_with_replacement_deduplicate(state **H, list *T);
__device__ void swap(state **s1, state **s2);
__device__ bool cuda_str_eq(const char *s1, const char *s2);
__device__ int cuda_atoi(const char *str);

heap **heaps_create(int k);
void heaps_destroy(heap **Q, int k);
char **expand_buf_create(int elements, int element_size);
__device__ bool heaps_empty(heap **heaps, int k);
__device__ int heaps_min(heap **heaps, int k);

state *states_pool_create();
void states_pool_destroy(state *states_pull);
__device__ state *state_create(const char *node, int f, int g, state *prev,
		state *states_pool);

__device__ void print_expanded(char **expanded) {
	for (int i = 0; expanded[i] != NULL; i++) {
		printf("%s\n", expanded[i]);
	}
}


#define STATES (1024 * 1024)
#define HASH_SIZE  (1024 * 1024)
#define HASH_FUNS 128

#define SLIDING_N  5
#define SLIDING_STATE_LEN (SLIDING_N * SLIDING_N)
#define SLIDING_EXPANDED_STATE_LEN (3 * SLIDING_STATE_LEN)

__device__ void expand_sliding(const char *str, char **result) {
	int len = 3 * SLIDING_STATE_LEN;
	int empty_pos;
	for (int i = 0; str[i] != '\0'; i++) {
		if (str[i] == '_') {
			empty_pos = i;
			break;
		}
	}
	empty_pos /= 3;
	int empty_row = empty_pos / SLIDING_N;
	int empty_col = empty_pos % SLIDING_N;
	int cur = 0;
	for (int new_row = empty_row - 1; new_row <= empty_row + 1; new_row++) {
		for (int new_col = empty_col - 1; new_col <= empty_col + 1; new_col++) {
			if (new_row < 0 || new_row >= SLIDING_N) continue;
			if (new_col < 0 || new_col >= SLIDING_N) continue;
			if (new_row != empty_row && new_col != empty_col) continue;
			if (new_row == empty_row && new_col == empty_col) continue;

			int new_pos = 3 * (SLIDING_N * new_row + new_col);
			result[cur] = (char*)malloc(3 * len + 1);
			memcpy(result[cur], str, SLIDING_EXPANDED_STATE_LEN + 1);
			result[cur][3 * empty_pos] = str[new_pos];
			result[cur][3 * empty_pos + 1] = str[new_pos + 1];
			result[cur][new_pos] = '_';
			result[cur][new_pos + 1] = '_';
			cur++;
		}
	}
	result[cur] = NULL;
}

__device__ int sliding_map[SLIDING_STATE_LEN + 1];

__device__ int h_sliding(const char *x, const char *t) {
	int res = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (x[3 * i] == '_') continue;
		int actual_row = i / SLIDING_N + 1;
		int actual_col = i % SLIDING_N + 1;
		int tile = cuda_atoi(&(x[3 * i]));
		int expected_row = sliding_map[tile] / SLIDING_N + 1;
		int expected_col = sliding_map[tile] % SLIDING_N + 1;
		res += abs(actual_row - expected_row) + abs(actual_col - expected_col);
	}
	return res;
}

__device__ expand_fun expand_sliding_gpu = expand_sliding;
__device__ heur_fun h_sliding_gpu = h_sliding;

int astar_gpu(const char *s_in, const char *t_in, int k) {

	char *s_gpu, *t_gpu;
	int map_cpu[SLIDING_STATE_LEN + 1];

	char *s_cpu = (char*)malloc(SLIDING_EXPANDED_STATE_LEN + 1);
	char *t_cpu = (char*)malloc(SLIDING_EXPANDED_STATE_LEN + 1);
	int ptr = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (s_in[ptr] == '_') {
			sprintf(s_cpu + 3 * i, "__,");
		} else {
			int current_s = atoi(&(s_in[ptr]));
			sprintf(s_cpu + 3 * i, "%02d,", current_s);
		}
		if (s_in[ptr+1] == ',') ptr += 2;
		else ptr += 3;
	}
	ptr = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (t_in[ptr] == '_') {
			sprintf(t_cpu + 3 * i, "__,");
		} else {
			int current_t = atoi(&(t_in[ptr]));
			map_cpu[current_t] = i;
			sprintf(t_cpu + 3 * i, "%02d,", current_t);
		}
		if (t_in[ptr+1] == ',') ptr += 2;
		else ptr += 3;
	}
	HANDLE_RESULT(cudaMalloc(&s_gpu, SLIDING_EXPANDED_STATE_LEN + 1));
	HANDLE_RESULT(cudaMalloc(&t_gpu, SLIDING_EXPANDED_STATE_LEN + 1));
	HANDLE_RESULT(cudaMemcpy(s_gpu, s_cpu, SLIDING_EXPANDED_STATE_LEN + 1, cudaMemcpyDefault));
	HANDLE_RESULT(cudaMemcpy(t_gpu, t_cpu, SLIDING_EXPANDED_STATE_LEN + 1, cudaMemcpyDefault));
	HANDLE_RESULT(cudaMemcpyToSymbol(sliding_map, map_cpu, (SLIDING_STATE_LEN + 1) * sizeof(int)));

	expand_fun expand_fun_cpu;
	HANDLE_RESULT(cudaMemcpyFromSymbol(&expand_fun_cpu, expand_sliding_gpu,
				sizeof(expand_fun)));
	heur_fun h_cpu;
	HANDLE_RESULT(cudaMemcpyFromSymbol(&h_cpu, h_sliding_gpu, sizeof(heur_fun)));

	state **H;
	char **expand_buf = expand_buf_create(4, SLIDING_EXPANDED_STATE_LEN);
	HANDLE_RESULT(cudaMalloc((void***)&H, HASH_SIZE * sizeof(state*)));
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
		swap(&t, &(H[index]));
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

__device__ void swap(state **s1, state **s2) {
	state *tmp = *s1;
	*s1 = *s2;
	*s2 = tmp;
}

__device__ unsigned int jenkins_hash(int j, const char *str) {
	char c;
	unsigned long hash = (j * 10000007);
	while (c = *str++) {
		hash += c;
		hash += hash << 10;
		hash ^= hash >> 6;
	}
	hash += hash << 3;
	hash ^= hash >> 11;
	hash += hash << 15;
	return hash;
}

heap **heaps_create(int k) {
	heap **Q_cpu = (heap**)malloc(k * sizeof(heap*));
	heap **Q_dev = NULL;
	for (int i = 0; i < k; i++) {
		Q_cpu[i] = heap_create(10000000);
	}
	HANDLE_RESULT(cudaMalloc(&Q_dev, k * sizeof(heap*)));
	HANDLE_RESULT(cudaMemcpy(Q_dev, Q_cpu, k * sizeof(heap*), cudaMemcpyDefault));
	free(Q_cpu);
	return Q_dev;
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

__device__ bool heaps_empty(heap **heaps, int k) {
	for (int i = 0; i < k; i++) {
		if (heaps[i]->size != 0) return false;
	}
	return true;
}

__device__ int heaps_min(heap **heaps, int k) {
	int best_f = INT_MAX;
	for (int i = 0; i < k; i++) {
		state *current_best = heaps[i]->states[0];
		if (current_best != NULL && current_best->f < best_f) {
			best_f = current_best->f;
		}
	}
	return best_f;
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

__device__ bool cuda_str_eq(const char *s1, const char *s2) {
	while(*s1) {
		if (*s1 != *s2) {
			return false;
		}
		s1++;
		s2++;
	}
	return true;
}

__device__ int cuda_atoi(const char *str) {
	int res = 0;
	while (*str >= '0' && *str <= '9') {
		res *= 10;
		res += *str - '0';
		str++;
	}
	return res;
}
