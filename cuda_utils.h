#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#define HANDLE_RESULT(expr) {cudaError_t _asdf__err; if ((_asdf__err = expr) != cudaSuccess) { printf("cuda call failed at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_asdf__err)); exit(1);}}

__device__ bool cuda_str_eq(const char *s1, const char *s2);

__device__ int cuda_atoi(const char *str);

__device__ unsigned int jenkins_hash(int j, const char *str);

#endif
