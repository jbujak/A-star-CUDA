#include "list.h"
#include <assert.h>
#include <stdio.h>

list *list_create(int capacity) {
	list *list = (struct list*)malloc(sizeof(struct list));
	list->length = 0;
	list->capacity = capacity;
	list->arr = (state**)calloc(capacity + 1, sizeof(state*));
	return list;
}

void list_clear(list *list) {
	list->length = 0;
}

void list_insert(list *list, state *state) {
	assert(list->length < list->capacity);
	list->arr[list->length++] = state;
}

void list_remove(list *list, int index) {
	assert(list->length < list->capacity);
	list->arr[index] = NULL;
}

state *list_get(list *list, int index) {
	assert(index < list->length);
	return list->arr[index];
}

void list_copy(list *dst, list *src) {
	list_clear(dst);
	for (int i = 0; i < src->length; i++) {
		list_insert(dst, list_get(src, i));
	}
}

int list_find(list *list, state *state) {
	for (int i = 0; i < list->length; i++) {
		if (strcmp(list_get(list, i)->node, state->node) == 0) {
			return i;
		}
	}
	return -1;
}

void list_destroy(list *list) {
	free(list->arr);
	free(list);
}
