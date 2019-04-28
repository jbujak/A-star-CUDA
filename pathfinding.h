#ifndef PATHFINDING_H
#define PATHFINDING_H
#include <fstream>
#include <string>

#include "astar_gpu.h"

void pathfinding_read_input(std::ifstream &file, std::string &s_out, std::string &t_out);

void pathfinding_preprocessing(const char *s_in, const char *t_in, char **s_out, char **t_out,
		expand_fun *expand_out, heur_fun *h_out, states_delta_fun *states_delta_out,
		int *expand_elements_out, int *expand_element_size_out);

#endif
