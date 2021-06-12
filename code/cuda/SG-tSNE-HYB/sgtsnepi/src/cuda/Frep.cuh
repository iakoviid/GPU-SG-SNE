
#ifndef FREP_CUH
#define FREP_CUH
#include "common.cuh"
#include "nuconv.cuh"
//#include "relocateData.cuh"
//#include "utils.cuh"
#include "utils_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

double computeFrepulsive_interpGPU(double *Freph, double *yh, int n,
                                   int d, double h, double *timeInfo);

#endif
