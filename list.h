#ifndef LIST_H
#define LIST_H

#include "astar_gpu.h"

struct list {
	int length;
	int capacity;
	state **arr;
};

list *list_create(int capacity);

void list_clear(list *list);

void list_insert(list *list, state *state);

void list_remove(list *list, int index);

state *list_get(list *list, int index);

void list_copy(list *dst, list *src);

int list_find(list *list, state *state);

void list_destroy(list *list);

#endif
