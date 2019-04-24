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

state *list_get(list *list, int index);

void list_destroy(list *list);

#endif
