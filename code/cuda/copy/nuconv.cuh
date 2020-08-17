#ifndef NUCONV_CUH
#define NUCONV_CUH
#define GRID_SIZE_THRESHOLD 100  // Differenet parallelism strategy for small grids
#include "common.cuh"
#include "gridding.cuh"
#include "non_periodic_conv.cuh"
#include <thrust/device_vector.h>

void nuconv(coord *PhiScat, coord *y, coord *VScat, uint32_t *ib, uint32_t *cb,
            int n, int d, int m, int nGridDim);
#endif
