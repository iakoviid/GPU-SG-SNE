#ifndef NUCONV_CUH
#define NUCONV_CUH
#define GRID_SIZE_THRESHOLD 600  // Differenet parallelism strategy for large grids
#include "common.cuh"
#include "gridding.cuh"
#include "non_periodic_convD.cuh"
#include <thrust/device_vector.h>

void nuconv(coord *PhiScat, coord *y, coord *VScat, int *ib,
            int n, int d, int m, int nGridDim,double * timeInfo);
#endif
