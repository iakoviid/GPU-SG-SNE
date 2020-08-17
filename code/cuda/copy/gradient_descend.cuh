
#ifndef GRADIENT__CUH
#define GRADIENT__CUH
#include "common.cuh"
#include "tsne.cuh"
#include "utils.cuh"
#include "Frep.cuh"
#include  "utils.cuh"
#include "utils_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/reduce.h>
template <class dataPoint>
double compute_gradient(dataPoint *dy, double *timeFrep, double *timeFattr,tsneparams params, dataPoint *y);
void kl_minimization(coord *y, tsneparams params);
#endif
