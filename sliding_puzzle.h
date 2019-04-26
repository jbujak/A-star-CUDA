#ifndef SLIDING_PUZZLE_H
#define SLIDING_PUZZLE_H

#include "astar_gpu.h"

void sliding_puzzle_preprocessing(const char *s_in, const char *t_in, char **s_out, char **t_out,
		expand_fun *expand_out, heur_fun *h_out,
		int *expand_elements_out, int *expand_element_size_out);

#endif
