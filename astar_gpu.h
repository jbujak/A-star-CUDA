#ifndef ASTAR_GPU
#define ASTAR_GPU

typedef char**(*expand_fun)(const char*);
int astar_cpu(const char *s, const char *t, int, expand_fun expand);

struct state {
	char *node;
	int f;
	int g;
	state *prev;
};

#endif //ASTAR_GPU
