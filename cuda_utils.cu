#include "cuda_utils.h"

__device__ bool cuda_str_eq(const char *s1, const char *s2) {
	while(*s1) {
		if(!*s1) break;
		if(!*s2) break;
		if (*s1 != *s2) {
			return false;
		}
		s1++;
		s2++;
	}
	return *s1 == *s2;
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

__device__ int cuda_strlen(const char *str) {
	int res = 0;
	while (*str++) res++;
	return res;
}

__device__ int cuda_sprintf_int(char* str, int n) {
	int _n = n;
	int len = 0;
	if (n == 0) {
		*str = '0';
		*(str+1) = '\0';
		return 1;
	}
	while(_n > 0) {
		_n /= 10;
		len++;
	}
	_n = n;
	int cur = len-1;
	while (_n > 0) {
		str[cur] = '0' + (_n % 10);
		_n /= 10;
		cur--;
	}
	str[len] = '\0';
	return len;
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

