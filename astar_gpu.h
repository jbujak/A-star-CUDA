#ifndef ASTAR_GPU
#define ASTAR_GPU

struct state;

typedef char**(*expand_fun)(const char*);
typedef int(*heur_fun)(const char *x, const char *t);
int astar_cpu(const char *s, const char *t, int k, expand_fun expand, heur_fun h);

struct state {
	const char *node;
	int f;
	int g;
	state *prev;
};

#endif //ASTAR_GPU
