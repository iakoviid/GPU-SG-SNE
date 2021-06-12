#ifndef GRIDDING_CUH
#define GRIDDING_CUH
#include "common.cuh"
#define LAGRANGE_INTERPOLATION
#ifdef LAGRANGE_INTERPOLATION
#define GRID_SIZE_THRESHOLD                                                    \
  600 // Differenet parallelism strategy for large grids
template<class dataPoint>
__inline__ __host__ __device__ dataPoint g1(dataPoint d) {
  return 0.5 * d * d * d - 1.0 * d * d - 0.5 * d + 1;
}
template<class dataPoint>
__inline__ __host__ __device__ dataPoint g2(dataPoint d) {
  coord cc = 1.0 / 6.0;
  return -cc * d * d * d + 1.0 * d * d - 11 * cc * d + 1;
}

#else
template<class dataPoint>
__inline__ __host__ __device__ dataPoint g1(dataPoint d) {
  return 1.5 * d * d * d - 2.5 * d * d + 1;
}
template<class dataPoint>
__inline__ __host__ __device__ dataPoint g2(dataPoint d) {
  return -0.5 * d * d * d + 2.5 * d * d - 4 * d + 2;
}

#endif
#define warpsize 32
#define BlockSizeWarp1D 32
#define BlockSizeWarp2D 32
#define BlockSizeWarp2Dshared 32

#define BlockSizeWarp3D 32
#define Gridsz 128
void s2g(coord __restrict__ *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
         uint32_t nDim, uint32_t nVec, int *ib);
void g2s(coord *PhiScat, coord *PhiGrid, coord *y, uint32_t nGridDim,
         uint32_t n, uint32_t d, uint32_t m);
void s2grb(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
           uint32_t ng, uint32_t n, uint32_t d, uint32_t nVec);

void s2gwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng, uint32_t n,
             uint32_t d, uint32_t nVec);
__global__ void s2g1d(coord __restrict__ *V, const coord *const y,
                      const coord *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec);
template <class dataPoint>
__global__ void g2s1d(dataPoint *Phi, dataPoint *V, dataPoint *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec);
__global__ void s2g2d(coord __restrict__ *V, const coord *const y,
                      const coord *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec);
__global__ void g2s2d(coord __restrict__ *Phi, const coord *const V,
                      const coord *const y, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec);
__global__ void s2g1drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec);
__global__ void s2g2drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec);
__global__ void s2g3d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec);
__global__ void g2s3d(coord __restrict__ *Phi, const coord *const V,
                      const coord *const y, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec);
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
