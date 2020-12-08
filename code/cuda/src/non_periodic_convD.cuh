#ifndef NUFFT_CUH
#define NUFFT_CUH
#include "common.cuh"
#include "complexD.cuh"
#include "cuComplex.h"
#include <cufft.h>
#include <cufftXt.h>

#ifndef KERNELS
#define KERNELS
__inline__ __device__ __host__ coord kernel1d(coord hsq, coord i) {
  return pow(1.0 + hsq * i * i, -2);
}

__inline__ __device__ __host__ coord kernel2d(coord hsq, coord i, coord j) {
  return pow(1.0 + hsq * (i * i + j * j), -2);
}

__inline__ __device__ __host__ coord kernel3d(coord hsq, coord i, coord j,
                                              coord k) {
  return pow(1.0 + hsq * (i * i + j * j + k * k), -2);
}
#endif
void convnopadCuda(coord *PhiGrid, coord *VGrid, coord h,
                   uint32_t *const nGridDims, int nVec, int nDim);
void conv1dnopadcuda(coord *PhiGrid, coord *VGrid, coord h,
                     uint32_t *const nGridDims, uint32_t nVec, int nDim);
void conv2dnopadcuda(coord *const PhiGrid, const coord *const VGrid,
                     const coord h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim);
void conv3dnopadcuda(coord *const PhiGrid, const coord *const VGrid,
                     const coord h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim);
#endif
