#include "gridding.cuh"
#include "matrix_indexing.hpp"
#include "utils_gpu.cuh"
#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx4(i, j, k, l, m, n, o) (SUB2IND4D(i, j, k, l, m, n, o))

#define shared true

void s2gwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng, uint32_t n,
             uint32_t d, uint32_t nVec) {

  switch (d) {

  case 1:
    s2g1drbwarp<<<Gridsz, BlockSizeWarp1D>>>(V, y, q, ib, ng, n, d, nVec);
    break;
  case 2:
    if (shared) {
      s2g2drbwarpshared<<<Gridsz, BlockSizeWarp2Dshared>>>(V, y, q, ib, ng, n,
                                                           d, nVec);
    } else {
      s2g2drbwarp<<<Gridsz, BlockSizeWarp2D>>>(V, y, q, ib, ng, n, d, nVec);
    }
    break;
  case 3:
    if (shared) {
      s2g3drbwarp<<<Gridsz, BlockSizeWarp3D>>>(V, y, q, ib, ng, n, d, nVec);
    } else {
      s2g3drbwarpfull<<<Gridsz, BlockSizeWarp3D>>>(V, y, q, ib, ng, n, d, nVec);
    }

    break;
  }
}
void s2g(coord *V, coord *y, coord *q, uint32_t ng, uint32_t n, uint32_t d,
         uint32_t nVec) {

  switch (d) {

  case 1:
    s2g1d<<<64, 512>>>(V, y, q, ng, n, d, nVec);
    break;
  case 2:
    s2g2d<<<64, 512>>>(V, y, q, ng, n, d, nVec);
    break;
  case 3:
    s2g3d<<<64, 512>>>(V, y, q, ng, n, d, nVec);
    break;
  }
}
__global__ void s2g1d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec) {
  coord v1[4];
  uint32_t f1;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      coord qv = q[nPts * j + TID];
      for (int idx1 = 0; idx1 < 4; idx1++) {
        atomicAdd(&V[f1 + idx1 + j * ng], qv * v1[idx1]);
      }
    }
  }
}
__device__ __inline__ void    WarpReduce2V1D(coord * V,coord *partial_sum,uint32_t nVec,uint32_t ng,uint32_t lane,uint32_t i)
{ coord val=0;
   for (uint32_t j = 0; j < nVec; j++) {
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        val = warp_reduce(
            partial_sum[(idx1 * nVec + j) * BlockSizeWarp1D + threadIdx.x]);
        if (lane == 0) {
          atomicAdd(&V[i + idx1 + j * ng], val);
        }
      }
    }

}
__global__ void s2g1drbwarpOld(coord *V, coord *y, coord *q, uint32_t *ib,
                             uint32_t ng, uint32_t nPts,
                            uint32_t nDim, uint32_t nVec) {
  //shared memory array for partial sums 8 elements for each thread
  __shared__ coord partial_sum[4 * 2 * BlockSizeWarp1D];
  register uint32_t f1;
  coord d;
  coord v1[4];
  register uint32_t i;
  register uint32_t idx1;
  register uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t lane = tid % 32;
  uint32_t warpid = (uint32_t)tid / warpsize;
  for (i = warpid; i < ng - 3; i += gridDim.x * blockDim.x / warpsize) {

    for (int x = 0; x < 8; x++) {
      partial_sum[x * BlockSizeWarp1D + threadIdx.x] = 0;
    }
    //for each point within the interval
    for (int TID = lane; TID <(int)( ib[i + 1] - ib[i]); TID += warpsize) {

      f1 = (uint32_t)floor(y[ib[i] + TID]);
      d = y[ib[i] + TID] - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);
      //Make partial results
      for (uint32_t j = 0; j < nVec; j++) {
        coord qv = q[nPts * j + ib[i] + TID];
        for (idx1 = 0; idx1 < 4; idx1++) {
          partial_sum[(idx1 * nVec + j) * BlockSizeWarp1D + threadIdx.x] +=
              qv * v1[idx1];
        } // (idx1)

      } // (j)

    } // (TID)
    WarpReduce2V1D(V,partial_sum,nVec,ng,lane,f1);
  }
}
__global__ void s2g1drbwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng,
                            uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  __shared__ coord partial_sum[4 * 2 * BlockSizeWarp1D];
  register int f1;
  coord d;
  coord v1[4];
  register int i, j;
  register int idx1;
  register int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = tid % 32;
  int warpid = (int)tid / warpsize;
  coord val;
  for (i = warpid; i < ng - 3; i += gridDim.x * blockDim.x / warpsize) {
    int offset = ib[i];
    int length = ib[i + 1] - ib[i];
    if (offset < 0 || length <= 0) {
      continue;
    }

    // loop through all points inside box
    for (int x = 0; x < 8; x++) {
      partial_sum[x * BlockSizeWarp1D + threadIdx.x] = 0;
    }
    for (int TID = lane; TID < length; TID += warpsize) {

      f1 = (int)floor(y[offset + TID]);
      d = y[offset + TID] - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);
      for (j = 0; j < nVec; j++) {

        coord qv = q[nPts * j + offset + TID];

        for (idx1 = 0; idx1 < 4; idx1++) {
          partial_sum[(idx1 * nVec + j) * BlockSizeWarp1D + threadIdx.x] +=
              qv * v1[idx1];
        } // (idx1)

      } // (j)

    } // (k)
    for (j = 0; j < nVec; j++) {
      for (idx1 = 0; idx1 < 4; idx1++) {
        val = warp_reduce(
            partial_sum[(idx1 * nVec + j) * BlockSizeWarp1D + threadIdx.x]);
        if (lane == 0) {
          atomicAdd(&V[i + idx1 + j * ng], val);
        }
      }
    }
  }
}
__global__ void g2s1d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  coord v1[4];
  uint32_t f1;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {
      coord accum = 0;
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        // printf("CUDA, V[%d]=%lf\n",f1 + idx1 + j * ng,V[f1 + idx1 + j * ng]
        // );
        accum += V[f1 + idx1 + j * ng] * v1[idx1];
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}

__global__ void s2g2d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec) {
  coord v1[4];
  coord v2[4];
  uint32_t f1;
  uint32_t f2;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (coord)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {

      for (int idx2 = 0; idx2 < 4; idx2++) {
        coord qv = q[nPts * j + TID] * v2[idx2];

        for (int idx1 = 0; idx1 < 4; idx1++) {

          atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng],
                    qv * v1[idx1]);
        }
      }
    }
  }
}

__inline__ __device__ coord warpReduce2V2D(coord *partial_sum, coord *V,
                                           uint32_t f1, uint32_t f2,
                                           uint32_t nVec, uint32_t ng,
                                           uint32_t lane) {
  coord val;
  for (uint32_t iVec = 0; iVec < nVec; iVec++) {

    for (uint32_t idx2 = 0; idx2 < 4; idx2++) {
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        val = warp_reduce(
            partial_sum[(iVec * 4 * 4 + idx2 * 4 + idx1) * BlockSizeWarp2D +
                        threadIdx.x]);
        if (lane == 0) {
          atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + iVec * ng * ng], val);
        }
      }
    }
  }

  return val;
}

__global__ void s2g2drbwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng,
                            uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  __shared__ coord partial_sum[4 * 4 * 3 * BlockSizeWarp2D];
  register uint32_t f1, f2;
  coord d;
  coord v1[4], v2[4];
  register uint32_t idx1, idx2;
  register uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t lane = tid % 32;
  uint32_t warpid = (uint32_t)tid / warpsize;
  // for(uint32_t boxIdx=warpid; boxIdx<(ng-2)*(ng-2)-ng-2;boxIdx+= gridDim.x *
  // blockDim.x / 32){
  // for (uint32_t i = warpid; i < ng - 2; i += gridDim.x * blockDim.x / 32) {
  // for (uint32_t j = 0; j < ng - 2; j++) {
  for (int boxIdx = warpid; boxIdx < (ng - 3) * (ng - 3);
       boxIdx += Gridsz * BlockSizeWarp2D / warpsize) {
    if (ib[boxIdx] < 0) {
      continue;
    }

    // uint32_t boxIdx = i * (ng - 2) + j;

    for (int temp = 0; temp < 4 * 4 * 3; temp++) {
      partial_sum[temp * BlockSizeWarp2D + threadIdx.x] = 0;
    }
    for (int TID = lane; TID < (int)ib[boxIdx + 1] - ib[boxIdx];
         TID += warpsize) {

      f1 = (uint32_t)floor(y[ib[boxIdx] + TID]);
      d = y[ib[boxIdx] + TID] - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);

      f2 = (uint32_t)floor(y[ib[boxIdx] + TID + nPts]);
      d = y[ib[boxIdx] + TID + nPts] - (coord)f2;

      v2[0] = g2(1 + d);
      v2[1] = g1(d);
      v2[2] = g1(1 - d);
      v2[3] = g2(2 - d);
      for (uint32_t iVec = 0; iVec < nVec; iVec++) {

        for (idx2 = 0; idx2 < 4; idx2++) {
          coord qv = q[nPts * iVec + ib[boxIdx] + TID] * v2[idx2];
          for (idx1 = 0; idx1 < 4; idx1++) {
            partial_sum[(iVec * 4 * 4 + idx2 * 4 + idx1) * BlockSizeWarp2D +
                        threadIdx.x] += qv * v1[idx1];
          }
        }
      }
    }

    warpReduce2V2D(partial_sum, V, f1, f2, nVec, ng, lane);
  }
}

__inline__ __device__ coord warpReduce2V2Dshared(coord *partial_sum, coord *V,
                                                 uint32_t f1, uint32_t f2,
                                                 uint32_t iVec, uint32_t ng,
                                                 uint32_t lane) {
  coord val;

  for (uint32_t idx2 = 0; idx2 < 4; idx2++) {
    for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
      val = warp_reduce(
          partial_sum[(idx2 * 4 + idx1) * BlockSizeWarp2Dshared + threadIdx.x]);
      if (lane == 0) {
        atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + iVec * ng * ng], val);
      }
    }
  }

  return val;
}

__global__ void s2g2drbwarpshared(coord *V, coord *y, coord *q, int *ib,
                                  uint32_t ng, uint32_t nPts, uint32_t nDim,
                                  uint32_t nVec) {
  __shared__ coord partial_sum[4 * 4 * BlockSizeWarp2Dshared];
  register uint32_t f1, f2;
  coord d;
  coord v1[4], v2[4];
  register uint32_t idx1, idx2;
  register uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
  uint32_t lane = tid % 32;
  uint32_t warpid = (uint32_t)tid / warpsize;
  // warp for every grid box
  for (int boxIdx = warpid; boxIdx < (ng - 3) * (ng - 3);
       boxIdx += Gridsz * BlockSizeWarp2Dshared / warpsize) {
    if (ib[boxIdx] < 0) {
      continue;
    }

    for (uint32_t iVec = 0; iVec < nVec; iVec++) {

      for (int temp = 0; temp < 4 * 4; temp++) {
        partial_sum[temp * BlockSizeWarp2Dshared + threadIdx.x] = 0;
      }
      for (int TID = lane; TID < (int)ib[boxIdx + 1] - ib[boxIdx];
           TID += warpsize) {

        f1 = (uint32_t)floor(y[ib[boxIdx] + TID]);
        d = y[ib[boxIdx] + TID] - (coord)f1;

        v1[0] = g2(1 + d);
        v1[1] = g1(d);
        v1[2] = g1(1 - d);
        v1[3] = g2(2 - d);

        f2 = (uint32_t)floor(y[ib[boxIdx] + TID + nPts]);
        d = y[ib[boxIdx] + TID + nPts] - (coord)f2;

        v2[0] = g2(1 + d);
        v2[1] = g1(d);
        v2[2] = g1(1 - d);
        v2[3] = g2(2 - d);

        for (idx2 = 0; idx2 < 4; idx2++) {
          coord qv = q[nPts * iVec + ib[boxIdx] + TID] * v2[idx2];
          for (idx1 = 0; idx1 < 4; idx1++) {
            partial_sum[(idx2 * 4 + idx1) * BlockSizeWarp2Dshared +
                        threadIdx.x] += qv * v1[idx1];
          }
        }
      }
      warpReduce2V2Dshared(partial_sum, V, boxIdx % (ng - 3), boxIdx / (ng - 3),
                           iVec, ng, lane);
    }
  }
}

__global__ void s2g3d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec) {
  coord v1[4];
  coord v2[4];
  coord v3[4];
  uint32_t f1;
  uint32_t f2;
  uint32_t f3;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (coord)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    f3 = (uint32_t)floor(y[TID + 2 * nPts]);
    d = y[TID + 2 * nPts] - (coord)f3;
    v3[0] = g2(1 + d);
    v3[1] = g1(d);
    v3[2] = g1(1 - d);
    v3[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      for (int idx3 = 0; idx3 < 4; idx3++) {

        for (int idx2 = 0; idx2 < 4; idx2++) {
          coord qv = q[nPts * j + TID] * v2[idx2] * v3[idx3];

          for (int idx1 = 0; idx1 < 4; idx1++) {
            atomicAdd(&V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, j, ng, ng, ng)],
                      qv * v1[idx1]);
          }
        }
      }
    }
  }
}

__inline__ __device__ coord warpReduce2V3Dfull(coord *partial_sum, coord *V,
                                               uint32_t f1, uint32_t f2,
                                               uint32_t f3, uint32_t nVec,
                                               uint32_t ng, uint32_t lane) {
  coord val;
  for (uint32_t iVec = 0; iVec < nVec; iVec++) {

    for (int idx3 = 0; idx3 < 4; idx3++) {

      for (uint32_t idx2 = 0; idx2 < 4; idx2++) {
        for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
          val = warp_reduce(
              partial_sum[(iVec * 4 * 4 * 4 + idx3 * 4 * 4 + idx2 * 4 + idx1) *
                              BlockSizeWarp3D +
                          threadIdx.x]);
          if (lane == 0) {
            atomicAdd(
                &V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, iVec, ng, ng, ng)],
                val);
          }
        }
      }
    }
  }

  return val;
}

__global__ void s2g3drbwarpfull(coord *V, coord *y, coord *q, int *ib,
                                uint32_t ng, uint32_t nPts, uint32_t nDim,
                                uint32_t nVec) {
  __shared__ coord partial_sum[4 * 4 * 4 * 5 * BlockSizeWarp3D];
  register uint32_t f1, f2, f3;
  coord d;
  coord v1[4], v2[4], v3[4];
  register uint32_t idx1, idx2, idx3;
  register uint32_t tid = blockIdx.x * BlockSizeWarp3D + threadIdx.x;
  uint32_t lane = tid % warpsize;
  uint32_t warpid = (uint32_t)tid / warpsize;

  for (int boxIdx = warpid; boxIdx < (ng - 3) * (ng - 3) * (ng - 3);
       boxIdx += Gridsz * BlockSizeWarp3D / warpsize) {
    if (ib[boxIdx] < 0) {
      continue;
    }

    for (int temp = 0; temp < 4 * 4 * 4 * 5; temp++) {
      partial_sum[temp * BlockSizeWarp3D + threadIdx.x] = 0;
    }
    for (int TID = lane; TID < (int)ib[boxIdx + 1] - ib[boxIdx];
         TID += warpsize) {

      f1 = (uint32_t)floor(y[ib[boxIdx] + TID]);
      d = y[ib[boxIdx] + TID] - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);

      f2 = (uint32_t)floor(y[ib[boxIdx] + TID + nPts]);
      d = y[ib[boxIdx] + TID + nPts] - (coord)f2;

      v2[0] = g2(1 + d);
      v2[1] = g1(d);
      v2[2] = g1(1 - d);
      v2[3] = g2(2 - d);

      f3 = (uint32_t)floor(y[ib[boxIdx] + TID + 2 * nPts]);
      d = y[ib[boxIdx] + TID + 2 * nPts] - (coord)f3;
      v3[0] = g2(1 + d);
      v3[1] = g1(d);
      v3[2] = g1(1 - d);
      v3[3] = g2(2 - d);
      for (uint32_t iVec = 0; iVec < nVec; iVec++) {
        for (idx3 = 0; idx3 < 4; idx3++) {

          for (idx2 = 0; idx2 < 4; idx2++) {
            coord qv = q[nPts * iVec + ib[boxIdx] + TID] * v2[idx2] * v3[idx3];
            for (idx1 = 0; idx1 < 4; idx1++) {
              partial_sum[(iVec * 4 * 4 * 4 + idx3 * 4 * 4 + idx2 * 4 + idx1) *
                              BlockSizeWarp3D +
                          threadIdx.x] += qv * v1[idx1];
            }
          }
        }
      }
    }
    warpReduce2V3Dfull(partial_sum, V, f1, f2, f3, nVec, ng, lane);
  }
}

__inline__ __device__ coord warpReduce2V3D(coord *partial_sum, coord *V,
                                           uint32_t f1, uint32_t f2,
                                           uint32_t f3, uint32_t iVec,
                                           uint32_t ng, uint32_t lane) {
  coord val;
  for (int idx3 = 0; idx3 < 4; idx3++) {

    for (uint32_t idx2 = 0; idx2 < 4; idx2++) {
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        val = warp_reduce(
            partial_sum[(idx3 * 4 * 4 + idx2 * 4 + idx1) * BlockSizeWarp3D +
                        threadIdx.x]);
        if (lane == 0) {
          atomicAdd(&V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, iVec, ng, ng, ng)],
                    val);
        }
      }
    }
  }

  return val;
}

__global__ void s2g3drbwarp(coord *V, coord *y, coord *q, int *ib, uint32_t ng,
                            uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  __shared__ coord partial_sum[4 * 4 * 4 * BlockSizeWarp3D];
  register uint32_t f1, f2, f3;
  coord d;
  coord v1[4], v2[4], v3[4];
  register uint32_t idx1, idx2, idx3;
  register uint32_t tid = blockIdx.x * BlockSizeWarp3D + threadIdx.x;
  uint32_t lane = tid % warpsize;
  uint32_t warpid = (uint32_t)tid / warpsize;

  for (int boxIdx = warpid; boxIdx < (ng - 3) * (ng - 3) * (ng - 3);
       boxIdx += Gridsz * BlockSizeWarp3D / warpsize) {
    if (ib[boxIdx] < 0) {
      continue;
    }

    for (uint32_t iVec = 0; iVec < nVec; iVec++) {

      for (int temp = 0; temp < 4 * 4 * 4; temp++) {
        partial_sum[temp * BlockSizeWarp3D + threadIdx.x] = 0;
      }
      for (int TID = lane; TID < (int)ib[boxIdx + 1] - ib[boxIdx];
           TID += warpsize) {

        f1 = (uint32_t)floor(y[ib[boxIdx] + TID]);
        d = y[ib[boxIdx] + TID] - (coord)f1;

        v1[0] = g2(1 + d);
        v1[1] = g1(d);
        v1[2] = g1(1 - d);
        v1[3] = g2(2 - d);

        f2 = (uint32_t)floor(y[ib[boxIdx] + TID + nPts]);
        d = y[ib[boxIdx] + TID + nPts] - (coord)f2;

        v2[0] = g2(1 + d);
        v2[1] = g1(d);
        v2[2] = g1(1 - d);
        v2[3] = g2(2 - d);

        f3 = (uint32_t)floor(y[ib[boxIdx] + TID + 2 * nPts]);
        d = y[ib[boxIdx] + TID + 2 * nPts] - (coord)f3;
        v3[0] = g2(1 + d);
        v3[1] = g1(d);
        v3[2] = g1(1 - d);
        v3[3] = g2(2 - d);

        for (idx3 = 0; idx3 < 4; idx3++) {

          for (idx2 = 0; idx2 < 4; idx2++) {
            coord qv = q[nPts * iVec + ib[boxIdx] + TID] * v2[idx2] * v3[idx3];
            for (idx1 = 0; idx1 < 4; idx1++) {
              partial_sum[(idx3 * 4 * 4 + idx2 * 4 + idx1) * BlockSizeWarp3D +
                          threadIdx.x] += qv * v1[idx1];
            }
          }
        }
      }
      warpReduce2V3D(partial_sum, V, f1, f2, f3, iVec, ng, lane);
    }
  }
}

__global__ void g2s2d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  coord v1[4];
  coord v2[4];
  uint32_t f1;
  uint32_t f2;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (coord)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      coord accum = 0;
      for (int idx2 = 0; idx2 < 4; idx2++) {
        coord qv = v2[idx2];

        for (int idx1 = 0; idx1 < 4; idx1++) {

          accum +=
              V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng] * qv * v1[idx1];
        }
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}
__global__ void g2s3d(coord *Phi, coord *V, coord *y, uint32_t ng,
                      uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  coord v1[4];
  coord v2[4];
  coord v3[4];
  uint32_t f1;
  uint32_t f2;
  uint32_t f3;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (coord)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    f3 = (uint32_t)floor(y[TID + 2 * nPts]);
    d = y[TID + 2 * nPts] - (coord)f3;
    v3[0] = g2(1 + d);
    v3[1] = g1(d);
    v3[2] = g1(1 - d);
    v3[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      coord accum = 0;
      for (int idx3 = 0; idx3 < 4; idx3++) {

        for (int idx2 = 0; idx2 < 4; idx2++) {
          coord qv = v2[idx2] * v3[idx3];

          for (int idx1 = 0; idx1 < 4; idx1++) {

            accum += V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, j, ng, ng, ng)] *
                     qv * v1[idx1];
          }
        }
        Phi[TID + j * nPts] = accum;
      }
    }
  }
}
