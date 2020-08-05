
#ifndef FREP_CUH
#define FREP_CUH
#include "common.cuh"
#include "nuconv.cuh"
#include "relocateData.cuh"
#include "utils.cuh"
#include "utils_cuda.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
coord computeFrepulsive_interp(coord *Frep, coord *y, int n, int d, coord h);
void ComputeCharges(coord *VScat, coord *y_d, int n, int d);
coord zetaAndForce(coord *Ft_d, coord *y_d, int n, int d, coord *Phi,
                   uint32_t *iPerm, thrust::device_vector<coord> &zetaVec);
#endif
