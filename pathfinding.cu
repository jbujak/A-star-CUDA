#include <fstream>
#include <string>
#include <iostream>

#include "pathfinding.h"
#include "astar_gpu.h"
#include "cuda_utils.h"

#define PATHFINDING_STATE_LEN (4+1+4) // "9999,9999"
#define BOARD_SIZE 10000

static int rows_cpu;
static int cols_cpu;

__device__ int rows_gpu;
__device__ int cols_gpu;

static int board_cpu[BOARD_SIZE][BOARD_SIZE];
__device__ int board_gpu[BOARD_SIZE][BOARD_SIZE];

void pathfinding_read_input(std::ifstream &file, std::string &s_out, std::string &t_out) {
	int o, w;

	file >> rows_cpu;
	file.ignore();
	file >> cols_cpu;
	file.ignore();

	std::getline(file, s_out);
	std::getline(file, t_out);

	file >> o;
	file.ignore();
	for (int i = 0; i < o; i++) {
		int x, y;
		file >> x;
		file.ignore();
		file >> y;
		board_cpu[x][y] = -1;
	}
	file >> w;
	file.ignore();
	for (int i = 0; i < w; i++) {
		int x, y, weight;
		file >> x;
		file.ignore();
		file >> y;
		file.ignore();
		file >> weight;
		board_cpu[x][y] = weight;
	}
}

__device__ void expand_pathfinding(const char *str, char **result) {
	int x = cuda_atoi(str);
	while(*str != ',')
		str++;
	str++;
	int y = cuda_atoi(str);
	int cur = 0;
	for (int i = -1; i <= 1; i++) {
		for (int j = -1; j <= 1; j++) {
			int len;
			if (i == 0 && j == 0) continue;
			if (x + i < 0 || x + i >= rows_gpu) continue;
			if (y + j < 0 || y + j >= cols_gpu) continue;
			if (board_gpu[x + i][y + j] == -1) continue;
			len = cuda_sprintf_int(result[cur], x + i);
			result[cur][len++] = ',';
			len = cuda_sprintf_int(result[cur] + len, y + j);
			cur++;
		}
	}
	result[cur][0] = '\0';
}

__device__ int h_pathfinding(const char *x, const char *t) {
	int x1, x2, y1, y2;
	x1 = cuda_atoi(x);
	x2 = cuda_atoi(t);

	while(*x != ',')
		x++;
	x++;
	while(*t != ',')
		t++;
	t++;

	y1 = cuda_atoi(x);
	y2 = cuda_atoi(t);

	return abs(x1 - x2) + abs(y1 - y2);
}

__device__ int states_delta_pathfinding(const char *src, const char *dst) {
	int x = cuda_atoi(dst);
	while(*dst != ',')
		dst++;
	dst++;
	int y = cuda_atoi(dst);
	int weight = board_gpu[x][y];
	return weight == 0 ? 1 : weight;
}

__device__ expand_fun expand_pathfinding_gpu = expand_pathfinding;
__device__ heur_fun h_pathfinding_gpu = h_pathfinding;
__device__ states_delta_fun states_delta_pathfinding_gpu = states_delta_pathfinding;

void pathfinding_preprocessing(const char *s_in, const char *t_in, char **s_out, char **t_out,
		expand_fun *expand_out, heur_fun *h_out, states_delta_fun *states_delta_out,
		int *expand_elements_out, int *expand_element_size_out) {
	char *s_gpu, *t_gpu;


	HANDLE_RESULT(cudaMalloc(&s_gpu, PATHFINDING_STATE_LEN + 1));
	HANDLE_RESULT(cudaMalloc(&t_gpu, PATHFINDING_STATE_LEN + 1));
	HANDLE_RESULT(cudaMemcpy(s_gpu, s_in, PATHFINDING_STATE_LEN + 1, cudaMemcpyDefault));
	HANDLE_RESULT(cudaMemcpy(t_gpu, t_in, PATHFINDING_STATE_LEN + 1, cudaMemcpyDefault));
	HANDLE_RESULT(cudaMemcpyToSymbol(board_gpu, board_cpu, BOARD_SIZE * BOARD_SIZE * sizeof(int)));
	HANDLE_RESULT(cudaMemcpyToSymbol(rows_gpu, &rows_cpu, sizeof(int)));
	HANDLE_RESULT(cudaMemcpyToSymbol(cols_gpu, &cols_cpu, sizeof(int)));

	expand_fun expand_fun_cpu;
	heur_fun h_cpu;
	states_delta_fun states_delta_cpu;
	HANDLE_RESULT(cudaMemcpyFromSymbol(&expand_fun_cpu, expand_pathfinding_gpu, sizeof(expand_fun)));
	HANDLE_RESULT(cudaMemcpyFromSymbol(&h_cpu, h_pathfinding_gpu, sizeof(heur_fun)));
	HANDLE_RESULT(cudaMemcpyFromSymbol(&states_delta_cpu, states_delta_pathfinding_gpu, sizeof(states_delta_fun)));

	*s_out = s_gpu;
	*t_out = t_gpu;
	*expand_out = expand_fun_cpu;
	*h_out = h_cpu;
	*states_delta_out = states_delta_cpu;
	*expand_elements_out = 9;
	*expand_element_size_out = PATHFINDING_STATE_LEN + 1;
}

