#include "cuda_utils.h"

__device__ bool cuda_str_eq(const char *s1, const char *s2) {
	while(*s1) {
		if (*s1 != *s2) {
			return false;
		}
		s1++;
		s2++;
	}
	return true;
}

__device__ int cuda_atoi(const char *str) {
	int res = 0;
	while (*str >= '0' && *str <= '9') {
		res *= 10;
		res += *str - '0';
		str++;
	}
	return res;
}

__device__ unsigned int jenkins_hash(int j, const char *str) {
	char c;
	unsigned long hash = (j * 10000007);
	while (c = *str++) {
		hash += c;
		hash += hash << 10;
		hash ^= hash >> 6;
	}
	hash += hash << 3;
	hash ^= hash >> 11;
	hash += hash << 15;
	return hash;
}

