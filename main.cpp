#include <iostream>

enum version_values {
	SLIDING, PATHFINDING
};

struct config {
	version_values version;
	std::string input_file;
	std::string output_file;
};

config parse_args(int argc, const char *argv[]);

int main(int argc, const char *argv[]) {
	config config;
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

