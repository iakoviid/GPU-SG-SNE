
#ifndef GRADIENT__CUH
#define GRADIENT__CUH
#include "Frep.cuh"
#include "common.cuh"
#include "pq.cuh"
#include "utils.cuh"
#include "utils_cuda.cuh"
#include <thrust/device_vector.h>
#include <sys/time.h>

#include <thrust/reduce.h>
void kl_minimization(double *y, tsneparams params, sparse_matrix<double> P);
void kl_minimization(float *y, tsneparams params, sparse_matrix<float> P,double* timeInfo);
#endif
