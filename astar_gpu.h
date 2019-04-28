#ifndef ASTAR_GPU
#define ASTAR_GPU

struct state;

enum version_value {
	SLIDING, PATHFINDING
};


typedef void(*expand_fun)(const char *x, char **result);
typedef int(*heur_fun)(const char *x, const char *t);
int astar_gpu(const char *s, const char *t, version_value version);

struct state {
	const char *node;
	int f;
	int g;
	state *prev;
};

#endif //ASTAR_GPU
