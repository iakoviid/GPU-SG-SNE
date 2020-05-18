#ifndef NUFFT_CUH
#define NUFFT_CUH
#include "common.cuh"
#include "complex.cuh"
#include <cufft.h>
#include <cufftXt.h>

#ifndef KERNELS
#define KERNELS
coord __device__ __host__ kernel1d(coord hsq, coord i) { return pow(1.0 + hsq * i * i, -2); }

coord __device__ __host__ kernel2d(coord hsq, coord i, coord j) {
  return pow(1.0 + hsq * (i * i + j * j), -2);
}

coord __device__ __host__ kernel3d(coord hsq, coord i, coord j, coord k) {
  return pow(1.0 + hsq * (i * i + j * j + k * k), -2);
}
#endif
void conv1dnopadcuda(coord *PhiGrid_d, coord *VGrid_d, coord h, int nGridDim,
                     int nVec, int nDim);
#endif
