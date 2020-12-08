#ifndef SPARSEMATRIX_CUH
#define SPARSEMATRIX_CUH
#include "types.hpp"
#include "common.cuh"
#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <cusparse.h>
#include "utils_gpu.cuh"

void free_sparse_matrixGPU(sparse_matrix * P);
uint32_t makeStochasticGPU(sparse_matrix *P);
void symmetrizeMatrixGPU(sparse_matrix*A,sparse_matrix *C);
void  permuteMatrixGPU(sparse_matrix *P, int *perm, int *iperm);
#endif
