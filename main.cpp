#include <iostream>

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

char** expand1(const char *str) {
	char **result = (char**)malloc(10 * sizeof(char*));
	result[0] = (char*)"a";
	result[1] = (char*)"b";
	result[2] = NULL;
	return result;
}

char** expand2(const char *str) {
	char **result = (char**)malloc(10 * sizeof(char*));
	result[0] = (char*)"x";
	result[1] = (char*)"y";
	result[2] = (char*)"z";
	result[3] = NULL;
	return result;
}

int main(int argc, const char *argv[]) {
	config config;
	astar_cpu("a", "b", 1, expand1);

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

