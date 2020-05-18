#ifndef NUCONV_HPP
#define NUCONV_HPP
#include <iostream>
#include <limits>
#include <cmath>

#include "timers.hpp"
#include "common.hpp"

void nuconvCPU( coord *PhiScat, coord *y, coord *VScat,
             uint32_t *ib, uint32_t *cb,
             int n, int d, int m, int np, int nGridDim);
#endif
