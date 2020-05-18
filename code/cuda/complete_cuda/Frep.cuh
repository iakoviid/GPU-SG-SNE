
#ifndef FREP_CUH
#define FREP_CUH
#include "common.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include "utils_cuda.cuh"
#include "utils.cuh"
#include "relocateData.cuh"
coord computeFrepulsive_interp(coord *Frep, coord *y, int n, int d, coord h);

#endif
