#ifndef NUFFT_CUH
#define NUFFT_CUH
#include "common.cuh"
#include "complex.cuh"
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
void conv1dnopadcuda(coord *PhiGrid_d, coord *VGrid_d, coord h, int nGridDim,
                     int nVec, int nDim);
void conv2dnopadcuda(double *const PhiGrid, const double *const VGrid,
                     const double h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim);
#endif
