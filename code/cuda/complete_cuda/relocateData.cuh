#ifndef DATARELOC_CUH
#define DATARELOC_CUH
#include "common.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

void relocateCoarseGrid(
    coord *Yptr,                            // Scattered point coordinates
    thrust::device_vector<uint32_t> &iPerm, // Data relocation permutation
    uint32_t *ib, // Starting index of box (along last dimension)
    uint32_t *cb, // Number of scattered points per box (along last dimension)
    int nPts,     // Number of data points
    int nGrid,    // Grid dimensions (+1)
    int nDim      // Number of dimensions
);

#endif
