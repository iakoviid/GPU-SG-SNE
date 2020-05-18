#ifndef DATARELOC_HPP
#define DATARELOC_HPP
#include <iostream>
#include <limits>
#include <cmath>
#include "common.hpp"
void relocateCoarseGridCPU( coord  ** Yptr,        // Scattered point coordinates
                         uint32_t ** iPermptr,    // Data relocation permutation
                         uint32_t *ib,            // Starting index of box (along last dimension)
                         uint32_t *cb,            // Number of scattered points per box (along last dimension)
                         int nPts,        // Number of data points
                         int nGridDim,    // Grid dimensions (+1)
                         int nDim,      // Number of dimensions
                         int np);
#endif
