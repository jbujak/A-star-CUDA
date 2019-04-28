#include "sliding_puzzle.h"
#include "cuda_utils.h"
#include <stdio.h>

#define SLIDING_N  5
#define SLIDING_STATE_LEN (SLIDING_N * SLIDING_N)
#define SLIDING_EXPANDED_STATE_LEN (3 * SLIDING_STATE_LEN)

__device__ void expand_sliding(const char *str, char **result) {
	int len = 3 * SLIDING_STATE_LEN;
	int empty_pos;
	for (int i = 0; str[i] != '\0'; i++) {
		if (str[i] == '_') {
			empty_pos = i;
			break;
		}
	}
	empty_pos /= 3;
	int empty_row = empty_pos / SLIDING_N;
	int empty_col = empty_pos % SLIDING_N;
	int cur = 0;
	for (int new_row = empty_row - 1; new_row <= empty_row + 1; new_row++) {
		for (int new_col = empty_col - 1; new_col <= empty_col + 1; new_col++) {
			if (new_row < 0 || new_row >= SLIDING_N) continue;
			if (new_col < 0 || new_col >= SLIDING_N) continue;
			if (new_row != empty_row && new_col != empty_col) continue;
			if (new_row == empty_row && new_col == empty_col) continue;

			int new_pos = 3 * (SLIDING_N * new_row + new_col);
			memcpy(result[cur], str, SLIDING_EXPANDED_STATE_LEN + 1);
			result[cur][3 * empty_pos] = str[new_pos];
			result[cur][3 * empty_pos + 1] = str[new_pos + 1];
			result[cur][new_pos] = '_';
			result[cur][new_pos + 1] = '_';
			cur++;
		}
	}
	result[cur][0] ='\0';
}

__device__ int sliding_map[SLIDING_STATE_LEN + 1];

__device__ int h_sliding(const char *x, const char *t) {
	int res = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (x[3 * i] == '_') continue;
		int actual_row = i / SLIDING_N + 1;
		int actual_col = i % SLIDING_N + 1;
		int tile = cuda_atoi(&(x[3 * i]));
		int expected_row = sliding_map[tile] / SLIDING_N + 1;
		int expected_col = sliding_map[tile] % SLIDING_N + 1;
		res += abs(actual_row - expected_row) + abs(actual_col - expected_col);
	}
	return res;
}

__device__ expand_fun expand_sliding_gpu = expand_sliding;
__device__ heur_fun h_sliding_gpu = h_sliding;

void sliding_puzzle_preprocessing(const char *s_in, const char *t_in, char **s_out, char **t_out,
		expand_fun *expand_out, heur_fun *h_out,
		int *expand_elements_out, int *expand_element_size_out) {
	char *s_gpu, *t_gpu;
	int map_cpu[SLIDING_STATE_LEN + 1];

	char *s_cpu = (char*)malloc(SLIDING_EXPANDED_STATE_LEN + 1);
	char *t_cpu = (char*)malloc(SLIDING_EXPANDED_STATE_LEN + 1);
	int ptr = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (s_in[ptr] == '_') {
			sprintf(s_cpu + 3 * i, "__,");
		} else {
			int current_s = atoi(&(s_in[ptr]));
			sprintf(s_cpu + 3 * i, "%02d,", current_s);
		}
		if (s_in[ptr+1] == ',') ptr += 2;
		else ptr += 3;
	}
	ptr = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (t_in[ptr] == '_') {
			sprintf(t_cpu + 3 * i, "__,");
		} else {
			int current_t = atoi(&(t_in[ptr]));
			map_cpu[current_t] = i;
			sprintf(t_cpu + 3 * i, "%02d,", current_t);
		}
		if (t_in[ptr+1] == ',') ptr += 2;
		else ptr += 3;
	}
	HANDLE_RESULT(cudaMalloc(&s_gpu, SLIDING_EXPANDED_STATE_LEN + 1));
	HANDLE_RESULT(cudaMalloc(&t_gpu, SLIDING_EXPANDED_STATE_LEN + 1));
	HANDLE_RESULT(cudaMemcpy(s_gpu, s_cpu, SLIDING_EXPANDED_STATE_LEN + 1, cudaMemcpyDefault));
	HANDLE_RESULT(cudaMemcpy(t_gpu, t_cpu, SLIDING_EXPANDED_STATE_LEN + 1, cudaMemcpyDefault));
	HANDLE_RESULT(cudaMemcpyToSymbol(sliding_map, map_cpu, (SLIDING_STATE_LEN + 1) * sizeof(int)));

	expand_fun expand_fun_cpu;
	HANDLE_RESULT(cudaMemcpyFromSymbol(&expand_fun_cpu, expand_sliding_gpu,
				sizeof(expand_fun)));
	heur_fun h_cpu;
	HANDLE_RESULT(cudaMemcpyFromSymbol(&h_cpu, h_sliding_gpu, sizeof(heur_fun)));

	*s_out = s_gpu;
	*t_out = t_gpu;
	*expand_out = expand_fun_cpu;
	*h_out = h_cpu;
	*expand_elements_out = 5;
	*expand_element_size_out = SLIDING_EXPANDED_STATE_LEN + 1;
}

std::string sliding_puzzle_postprocessing(std::vector<std::string> in) {
	std::string result;
	for (std::string line: in) {
		for (int i = 0; i < line.length(); i += 3) {
			if (line[i] != '0' && line[i] != '_') result += line[i];
			result += line[i + 1];
			result += ",";
		}
		result[result.length() - 1] = '\n';
	}
	return result;
}
