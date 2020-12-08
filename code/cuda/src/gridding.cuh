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
#define warpsize 32
#define BlockSizeWarp1D 32
#define BlockSizeWarp2D 32
#define BlockSizeWarp2Dshared 32

#define BlockSizeWarp3D 32
#define Gridsz 128
void s2grb(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
           uint32_t ng, uint32_t n, uint32_t d, uint32_t nVec);
void s2g(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
         uint32_t nDim, uint32_t nVec);
void s2gwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng, uint32_t n,
             uint32_t d, uint32_t nVec);
__global__ void s2g1d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec);
__global__ void g2s1d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g2d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec);
__global__ void g2s2d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g1drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec);
__global__ void s2g2drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec);
__global__ void s2g3d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec);
__global__ void g2s3d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g3drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec);
__global__ void s2g1drbwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng,
                            uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g2drbwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng,
                            uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g3drbwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng,
                            uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g2drbwarpshared(coord *V, coord *y, coord *q, int *ib,
                                  uint32_t ng, uint32_t nPts, uint32_t nDim,
                                  uint32_t nVec);
__global__ void s2g3drbwarpfull(coord *V, coord *y, coord *q, int *ib,
                                uint32_t ng, uint32_t nPts, uint32_t nDim,
                                uint32_t nVec);
__global__ void s2g1drbwarpOld(coord *V, coord *y, coord *q, uint32_t *ib,
                               uint32_t ng, uint32_t nPts, uint32_t nDim,
                               uint32_t nVec);
#endif
