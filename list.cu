#include "list.h"
#include <assert.h>

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

state *list_get(list *list, int index) {
	assert(index < list->length);
	return list->arr[index];
}

void list_destroy(list *list) {
	free(list->arr);
	free(list);
}
