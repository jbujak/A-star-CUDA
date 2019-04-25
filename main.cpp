#include <iostream>
#include <string.h>

#include "astar_gpu.h"

enum version_values {
	SLIDING, PATHFINDING
};

struct config {
	version_values version;
	std::string input_file;
	std::string output_file;
};

config parse_args(int argc, const char *argv[]);

char** expand(const char *str) {
	int node = atoi(str);
	char **result = (char**)malloc(4 * sizeof(char*));
	char *arr = (char*)malloc(3 * 11 * sizeof(char*));
	int sibling = node + 1;
	int lchild = 2 * node;
	int rchild = 2 * node + 1;
	sprintf(arr, "%04d %04d %04d ", sibling, lchild, rchild);
	for (int i = 0; i < 3; i++) {
		result[i] = &(arr[5 * i]);
		arr[5 * i - 1] = '\0';
	}
	if (node == 1) {
		result[2] = NULL;
	}
	result[3] = NULL;
	return result;
}

int h(const char *x, const char *t) {
	int res = 0;
	if (atoi(x) > atoi(t)) return 20000;
	int dist = abs(atoi(x) - atoi(t));
	while (dist > 0) {
		dist /= 2;
		res++;
	}
	return res;
}

int map[10];
int first_char[10];

const int SLIDING_N = 5;
const int SLIDING_STATE_LEN = SLIDING_N * SLIDING_N;

char** expand_sliding(const char *str) {
	int len = 3 * SLIDING_STATE_LEN;
	char **result = (char**)malloc(5 * sizeof(char*));
	int empty_pos = (strchr(str, '_') - str) / 3;
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
			result[cur] = (char*)malloc(3 * len + 1);
			strcpy(result[cur], str);
			result[cur][3 * empty_pos] = str[new_pos];
			result[cur][3 * empty_pos + 1] = str[new_pos + 1];
			result[cur][new_pos] = '_';
			result[cur][new_pos + 1] = '_';
			cur++;
		}
	}
	result[cur] = NULL;
	return result;
}

int h_sliding(const char *x, const char *t) {
	int res = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (x[3 * i] == '_') continue;
		int actual_row = i / SLIDING_N + 1;
		int actual_col = i % SLIDING_N + 1;
		int tile = atoi(&(x[3 * i]));
		int expected_row = map[tile] / SLIDING_N + 1;
		int expected_col = map[tile] % SLIDING_N + 1;
		res += abs(actual_row - expected_row) + abs(actual_col - expected_col);
	}
	return res;
}

int main(int argc, const char *argv[]) {
	config config;

	//const char *s_in = "2,1,3,5,_,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24";
	//const char *t_in = "2,1,_,3,5,4,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24";
	const char *s_in = "1,2,_,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24";
	const char *t_in = "4,9,8,12,14,_,18,13,11,19,21,23,20,24,15,1,3,7,16,5,6,2,10,17,22";

	char *s = (char*)malloc(30 * 3 * sizeof(char));
	char *t = (char*)malloc(30 * 3 * sizeof(char));
	
	int ptr = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (s_in[ptr] == '_') {
			sprintf(s + 3 * i, "__,");
		} else {
			int current_s = atoi(&(s_in[ptr]));
			sprintf(s + 3 * i, "%02d,", current_s);
		}
		if (s_in[ptr+1] == ',') ptr += 2;
		else ptr += 3;
	}
	ptr = 0;
	for (int i = 0; i < SLIDING_STATE_LEN; i++) {
		if (t_in[ptr] == '_') {
			sprintf(t + 3 * i, "__,");
		} else {
			int current_t = atoi(&(t_in[ptr]));
			map[current_t] = i;
			sprintf(t + 3 * i, "%02d,", current_t);
		}
		if (t_in[ptr+1] == ',') ptr += 2;
		else ptr += 3;
	}
	astar_cpu(s, t, 1, expand_sliding, h_sliding);
	return 0;

	return 0;
	try {
		config = parse_args(argc, argv);
	} catch (std::string error) {
		std::cout << error << std::endl;
		return 1;
	}
	return 0;
}

std::string usage(std::string filename) {
	return "Usage: " + filename + " --version [sliding | pathfinding]" +
		" --input-data input.txt --output-data output.txt";
}

config parse_args(int argc, const char *argv[]) {
	config result = {};
	std::string filename = std::string(argv[0]);
	if (argc != 7) throw usage(filename);

	if (std::string(argv[1]) != "--version") throw usage(filename);
	std::string version = std::string(argv[2]);
	if (version == "sliding") result.version = SLIDING;
	else if (version == "pathfinding") result.version = PATHFINDING;
	else throw usage(filename);

	if (std::string(argv[3]) != "--input-data") throw usage(filename);
	result.input_file = std::string(argv[4]);

	if (std::string(argv[5]) != "--output-data") throw usage(filename);
	result.output_file = std::string(argv[6]);
	return result;
}

