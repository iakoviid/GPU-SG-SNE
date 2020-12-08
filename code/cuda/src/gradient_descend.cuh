
#ifndef GRADIENT__CUH
#define GRADIENT__CUH
#include "common.cuh"
#include "tsne.cuh"
#include "Frep.cuh"
#include  "utils.cuh"
#include "utils_cuda.cuh"
#include "pq.cuh"
#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "timer.h"
template <class dataPoint>
double compute_gradient(dataPoint *dy, double *timeFrep, double *timeFattr,
                        tsneparams params, dataPoint *y,sparse_matrix P);
void kl_minimization(coord *y, tsneparams params,sparse_matrix P);
#endif
