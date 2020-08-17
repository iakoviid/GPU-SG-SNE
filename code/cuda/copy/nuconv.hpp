#ifndef NUCONV_HPP
#define NUCONV_HPP
#include <iostream>
#include <limits>
#include <cmath>
#define GRID_SIZE_THRESHOLD 100  // Differenet parallelism strategy for small grids

#include "timers.hpp"
#include "common.hpp"

void nuconvCPU( coord *PhiScat, coord *y, coord *VScat,
             uint32_t *ib, uint32_t *cb,
             int n, int d, int m, int np, int nGridDim);
#endif
