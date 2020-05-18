#ifndef GRIDDING_CUH
#define GRIDDING_CUH
#include "common.cuh"
#ifdef LAGRANGE_INTERPOLATION

__inline__ __host__ __device__ coord g1(coord d) {
  return 0.5 * d * d * d - 1.0 * d * d - 0.5 * d + 1;
}

__inline__ __host__ __device__ coord g2(coord d) {
  coord cc = 1.0 / 6.0;
  return -cc * d * d * d + 1.0 * d * d - 11 * cc * d + 1;
}

#else

__inline__ __host__ __device__ coord g1(coord d) {
  return 1.5 * d * d * d - 2.5 * d * d + 1;
}

__inline__ __host__ __device__ coord g2(coord d) {
  return -0.5 * d * d * d + 2.5 * d * d - 4 * d + 2;
}

#endif
__global__ void s2g1d(coord *V, coord *y, coord *q, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec, coord maxy);
__global__ void g2s1d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec);
#endif
