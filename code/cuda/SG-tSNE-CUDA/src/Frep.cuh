
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

template <class dataPoint, class Complext>
dataPoint computeFrepulsive_interp(dataPoint *Frep, dataPoint *y, int n, int d,
                                   dataPoint h, double *timeInfo, int nGrid,
                                   cufftHandle &plan, cufftHandle &plan_rhs,
                                   dataPoint *yt, dataPoint *VScat,
                                   dataPoint *PhiScat, dataPoint *VGrid,
                                   dataPoint *PhiGrid, Complext *Kc,
                                   Complext *Xc,thrust::device_vector<dataPoint> &zetaVec);
#endif
