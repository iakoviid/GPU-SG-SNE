#ifndef NUFFT_HPP
#define NUFFT_HPP
#include "common.hpp"
#include "matrix_indexing.hpp"
#include <math.h>       /* pow */
#ifndef KERNELSCPU
#define KERNELSCPU
__inline__  coord kernel1dCPU(coord hsq, coord i) {
  return pow(1.0 + hsq * i * i, -2);
}

__inline__  coord kernel2dCPU(coord hsq, coord i, coord j) {
  return pow(1.0 + hsq * (i * i + j * j), -2);
}

__inline__  coord kernel3dCPU(coord hsq, coord i, coord j,
                                              coord k) {
  return pow(1.0 + hsq * (i * i + j * j + k * k), -2);
}
#endif

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
