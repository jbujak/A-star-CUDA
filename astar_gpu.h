#ifndef ASTAR_GPU
#define ASTAR_GPU

#define HANDLE_RESULT(expr) {cudaError_t _asdf__err; if ((_asdf__err = expr) != cudaSuccess) { printf("cuda call failed at line %d: %s\n", __LINE__, cudaGetErrorString(_asdf__err)); exit(1);}}

struct state;

typedef void(*expand_fun)(const char *x, char **result);
typedef int(*heur_fun)(const char *x, const char *t);
int astar_gpu(const char *s, const char *t, int k);

struct state {
	const char *node;
	int f;
	int g;
	state *prev;
};

#endif //ASTAR_GPU
