#ifndef DATARELOC_CUH
#define DATARELOC_CUH
#include "common.cuh"
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>

void relocateCoarseGrid(
    coord *Yptr,                            // Scattered point coordinates
    uint32_t* iPerm, // Data relocation permutation
    int *ib, // Starting index of box (along last dimension)
    int nPts,     // Number of data points
    int nGrid,    // Grid dimensions (+1)
    int nDim      // Number of dimensions
);

#endif
