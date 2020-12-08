#ifndef NUFFT_HPP
#define NUFFT_HPP
#include "common.hpp"
#include "matrix_indexing.hpp"
#include "non_periodic_conv.cuh"

#include <cmath>
#include <complex>
#include <fftw3.h>
#include <iostream>

void conv1dnopad(coord *const PhiGrid, const coord *const VGrid, const coord h,
                 uint32_t *const nGridDims, const uint32_t nVec,
                 const uint32_t nDim, const uint32_t nProc);
void conv2dnopad(coord *const PhiGrid, const coord *const VGrid, const coord h,
                 uint32_t *const nGridDims, const uint32_t nVec,
                 const uint32_t nDim, const uint32_t nProc);
void conv3dnopad(coord *const PhiGrid, const coord *const VGrid, const coord h,
                 uint32_t *const nGridDims, const uint32_t nVec,
                 const uint32_t nDim, const uint32_t nProc);
void connopad(coord *const PhiGrid, const coord *const VGrid, const coord h,
              uint32_t *const nGridDims, const uint32_t nVec,
              const uint32_t nDim, const uint32_t nProc);
#endif
