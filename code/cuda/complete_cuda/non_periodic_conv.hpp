#ifndef NUFFT_HPP
#define NUFFT_HPP
#include <iostream>
#include "common.hpp"
#include <complex>
#include <fftw3.h>
#include <cmath>
#include "matrix_indexing.hpp"
#include "non_periodic_conv.cuh"
void conv1dnopad(coord *const PhiGrid, const coord *const VGrid,
                 const coord h, uint32_t *const nGridDims, const uint32_t nVec,
                 const uint32_t nDim, const uint32_t nProc);
#endif
