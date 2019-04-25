#include <stdio.h>

#include "astar_gpu.h"
#include "heap.h"
#include "list.h"

int f(const state *x, const char *t, heur_fun h);
unsigned int jenkins_hash(int j, const char *str);
void hash_with_replacement_deduplicate(state **H, list *T);
void swap(state **s1, state **s2);

heap **heaps_create(int k);
bool heaps_empty(heap **heaps, int k);
int heaps_min(heap **heaps, int k);
void heaps_destroy(heap **Q, int k);

state *state_create(const char *node, int f, int g, state *prev);
void state_destroy(state *state);

void print_expanded(char **expanded) {
	for (int i = 0; expanded[i] != NULL; i++) {
		printf("%s\n", expanded[i]);
	}
}


#define HASH_SIZE  (1024 * 1024)
#define HASH_FUNS 128
int astar_cpu(const char *s, const char *t, int k, expand_fun expand, heur_fun h) {
	heap **Q = heaps_create(k);
	list *S = list_create(10000000);
	state **H = (state**)calloc(HASH_SIZE, sizeof(state*));
	state *m = NULL;

	heap_insert(Q[0], state_create(s, 0, 0, NULL));

	int steps = 0;
	while (!heaps_empty(Q, k)) {
		steps++;
		list_clear(S);
		for (int i = 0; i < k; i++) {
			if (Q[i]->size == 0) continue;
			state *q = heap_extract(Q[i]);
			//printf("%s\n", q->node);
			if (steps % 1000 == 0) printf("distance: %d\n", h(q->node, t));
			if (strcmp(q->node, t) == 0) {
				if (m == NULL || f(q, t, h) < f(m, t, h)) {
					m = q;
				}
				continue;
			}
			char **expanded = expand(q->node);
			for (int j = 0; expanded[j] != NULL; j++) {
				list_insert(S, state_create(expanded[j], -1, q->g + 1, q));
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

	free(H);
	list_destroy(S);
	heaps_destroy(Q, k);
	return 0;
}

void hash_with_replacement_deduplicate(state **H, list *T) {
	for (int i = 0; i < T->length; i++) {
		int z = 0;
		state *t = list_get(T, i);
		for (int j = 0; j < HASH_FUNS; j++) {
			state *el = H[jenkins_hash(j, t->node) % HASH_SIZE];
			if (el == NULL || strcmp(t->node, el->node) == 0) {
				z = j;
				break;
			}
		}
		int index = jenkins_hash(z, t->node) % HASH_SIZE;
		swap(&t, &(H[index]));
		if (t != NULL && strcmp(t->node, list_get(T, i)->node) == 0) {
			list_remove(T, i);
			continue;
		}
		t = list_get(T, i);
		for (int j = 0; j < HASH_FUNS; j++) {
			if (j != z) {
				state *el = H[jenkins_hash(j, t->node) % HASH_SIZE];
				if (el != NULL && strcmp(el->node, t->node) == 0) {
					list_remove(T, i);
					break;
				}
			}
		}
	}
}

int f(const state *x, const char *t, heur_fun h) {
	return x->g + h(x->node, t);
}

void swap(state **s1, state **s2) {
	state *tmp = *s1;
	*s1 = *s2;
	*s2 = tmp;
}

unsigned int jenkins_hash(int j, const char *str) {
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
	heap **Q = (heap**)malloc(k * sizeof(heap*));
	for (int i = 0; i < k; i++) {
		Q[i] = heap_create(10000000);
	}
	return Q;
}

bool heaps_empty(heap **heaps, int k) {
	for (int i = 0; i < k; i++) {
		if (heaps[i]->size != 0) return false;
	}
	return true;
}

int heaps_min(heap **heaps, int k) {
	int best_f = INT_MAX;
	for (int i = 0; i < k; i++) {
		state *current_best = heaps[i]->states[0];
		if (current_best != NULL && current_best->f < best_f) {
			best_f = current_best->f;
		}
	}
	return best_f;
}

void heaps_destroy(heap **Q, int k) {
	for (int i = 0; i < k; i++) {
		heap_destroy(Q[i]);
	}
	free(Q);
}

state *state_create(const char *node, int f, int g, state *prev) {
	state *result = (state*)malloc(sizeof(state));
	result->node = node;
	result->f = f;
	result->g = g;
	result->prev = prev;
	return result;
}

void state_destroy(state *state) {
	free(state);
}
