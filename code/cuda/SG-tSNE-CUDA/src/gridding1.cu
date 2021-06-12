#include "gridding1.cuh"
#include "matrix_indexing.hpp"
#include "utils_gpu.cuh"

#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx4(i, j, k, l, m, n, o) (SUB2IND4D(i, j, k, l, m, n, o))
template <class dataPoint>
void s2g(dataPoint *VGrid, dataPoint *y, dataPoint *VScat, uint32_t nGridDim,
         uint32_t n, uint32_t d, uint32_t m, int *ib) {
  switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD) {

      s2g1d<<<64, 1024>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);
    } else {
      s2g1drbwarp<<<Gridsz, BlockSizeWarp1D>>>(VGrid, y, VScat, ib,
                                               nGridDim + 2, n, d, m);
    }

    break;

  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD) {

      s2g2d<<<64, 1024>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);

    } else {

      s2g2drbwarpshared<<<Gridsz, BlockSizeWarp2Dshared>>>(
          VGrid, y, VScat, ib, nGridDim + 2, n, d, m);
    }
    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD) {
      s2g3d<<<64, 1024>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);
    } else {
      s2g3drbwarp<<<Gridsz, BlockSizeWarp3D>>>(VGrid, y, VScat, ib,
                                               nGridDim + 2, n, d, m);
    }
    break;
  }
}
template <class dataPoint>
void g2s(dataPoint *PhiScat, dataPoint *PhiGrid, dataPoint *y,
         uint32_t nGridDim, uint32_t n, uint32_t d, uint32_t m) {
  switch (d) {

  case 1:
    g2s1d<<<64, 1024>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);

    break;

  case 2:
    g2s2d<<<64, 1024>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;

  case 3:
    g2s3d<<<64, 1024>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;
  }
}

template <class dataPoint>
__global__ void s2g1d(dataPoint *__restrict__ V, const dataPoint *const y,
                      const dataPoint *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec) {
  dataPoint v1[4];
  register uint32_t f1;
  register dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (dataPoint)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      dataPoint qv = q[nPts * j + TID];
      for (int idx1 = 0; idx1 < 4; idx1++) {
        atomicAdd(&V[f1 + idx1 + j * ng], qv * v1[idx1]);
      }
    }
  }
}

template <class dataPoint>
__global__ void g2s1d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec) {
  dataPoint v1[4];
  uint32_t f1;
  dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (dataPoint)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {
      dataPoint accum = 0;
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        accum += V[f1 + idx1 + j * ng] * v1[idx1];
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}
template <class dataPoint>
__global__ void s2g2d(dataPoint *__restrict__ V, const dataPoint *const y,
                      const dataPoint *const q, const uint32_t ng,
                      const uint32_t nPts, const uint32_t nDim,
                      const uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  register uint32_t f1;
  register uint32_t f2;
  register dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (dataPoint)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (dataPoint)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {

      for (int idx2 = 0; idx2 < 4; idx2++) {
        dataPoint qv = q[nPts * j + TID] * v2[idx2];

        for (int idx1 = 0; idx1 < 4; idx1++) {

          atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng],
                    qv * v1[idx1]);
        }
      }
    }
  }
}
template <class dataPoint>
__global__ void g2s2d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  register uint32_t f1;
  register uint32_t f2;
  register dataPoint d;
  register dataPoint accum = 0;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (dataPoint)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (dataPoint)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      accum = 0;
      for (int idx2 = 0; idx2 < 4; idx2++) {
        dataPoint qv = v2[idx2];

        for (int idx1 = 0; idx1 < 4; idx1++) {

          accum +=
              V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng] * qv * v1[idx1];
        }
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}
template <class dataPoint>
__global__ void s2g3d(dataPoint *__restrict__ V, dataPoint *y, dataPoint *q,
                      uint32_t ng, uint32_t nPts, uint32_t nDim,
                      uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  dataPoint v3[4];
  uint32_t f1;
  uint32_t f2;
  uint32_t f3;
  dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (dataPoint)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (dataPoint)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    f3 = (uint32_t)floor(y[TID + 2 * nPts]);
    d = y[TID + 2 * nPts] - (dataPoint)f3;
    v3[0] = g2(1 + d);
    v3[1] = g1(d);
    v3[2] = g1(1 - d);
    v3[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      for (int idx3 = 0; idx3 < 4; idx3++) {

        for (int idx2 = 0; idx2 < 4; idx2++) {
          dataPoint qv = q[nPts * j + TID] * v2[idx2] * v3[idx3];

          for (int idx1 = 0; idx1 < 4; idx1++) {
            atomicAdd(&V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, j, ng, ng, ng)],
                      qv * v1[idx1]);
          }
        }
      }
    }
  }
}
template <class dataPoint>
__global__ void g2s3d(volatile dataPoint *__restrict__ Phi,
                      const dataPoint *const V, const dataPoint *const y,
                      const uint32_t ng, const uint32_t nPts,
                      const uint32_t nDim, const uint32_t nVec) {
  dataPoint v1[4];
  dataPoint v2[4];
  dataPoint v3[4];
  register uint32_t f1;
  register uint32_t f2;
  register uint32_t f3;
  register dataPoint d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (dataPoint)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y[TID + nPts]);
    d = y[TID + nPts] - (dataPoint)f2;
    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    f3 = (uint32_t)floor(y[TID + 2 * nPts]);
    d = y[TID + 2 * nPts] - (dataPoint)f3;
    v3[0] = g2(1 + d);
    v3[1] = g1(d);
    v3[2] = g1(1 - d);
    v3[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      dataPoint accum = 0;
      for (int idx3 = 0; idx3 < 4; idx3++) {

        for (int idx2 = 0; idx2 < 4; idx2++) {
          dataPoint qv = v2[idx2] * v3[idx3];

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
template <class dataPoint>
__global__ void s2g1drbwarp(dataPoint *V, dataPoint *y, dataPoint *q, int *ib,
                            uint32_t ng, uint32_t nPts, uint32_t nDim,
                            uint32_t nVec) {
  __shared__ dataPoint partial_sum[4 * 2 * BlockSizeWarp1D];
  register int f1;
  dataPoint d;
  dataPoint v1[4];
  register int i, j;
  register int idx1;
  register int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int lane = tid % 32;
  int warpid = (int)tid / warpsize;
  dataPoint val;
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
      d = y[offset + TID] - (dataPoint)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);
      for (j = 0; j < nVec; j++) {

        dataPoint qv = q[nPts * j + offset + TID];

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
template <class dataPoint>
__inline__ __device__ dataPoint warpReduce2V2Dshared(dataPoint *partial_sum,
                                                     dataPoint *V, uint32_t f1,
                                                     uint32_t f2, uint32_t iVec,
                                                     uint32_t ng,
                                                     uint32_t lane) {
  dataPoint val;

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
template <class dataPoint>
__global__ void s2g2drbwarpshared(dataPoint *V, dataPoint *y, dataPoint *q,
                                  int *ib, uint32_t ng, uint32_t nPts,
                                  uint32_t nDim, uint32_t nVec) {
  __shared__ dataPoint partial_sum[4 * 4 * BlockSizeWarp2Dshared];
  register uint32_t f1, f2;
  dataPoint d;
  dataPoint v1[4], v2[4];
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
        d = y[ib[boxIdx] + TID] - (dataPoint)f1;

        v1[0] = g2(1 + d);
        v1[1] = g1(d);
        v1[2] = g1(1 - d);
        v1[3] = g2(2 - d);

        f2 = (uint32_t)floor(y[ib[boxIdx] + TID + nPts]);
        d = y[ib[boxIdx] + TID + nPts] - (dataPoint)f2;

        v2[0] = g2(1 + d);
        v2[1] = g1(d);
        v2[2] = g1(1 - d);
        v2[3] = g2(2 - d);

        for (idx2 = 0; idx2 < 4; idx2++) {
          dataPoint qv = q[nPts * iVec + ib[boxIdx] + TID] * v2[idx2];
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
template <class dataPoint>
__inline__ __device__ dataPoint warpReduce2V3D(dataPoint *partial_sum,
                                               dataPoint *V, uint32_t f1,
                                               uint32_t f2, uint32_t f3,
                                               uint32_t iVec, uint32_t ng,
                                               uint32_t lane) {
  dataPoint val;
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
template <class dataPoint>
__global__ void s2g3drbwarp(dataPoint *V, dataPoint *y, dataPoint *q, int *ib,
                            uint32_t ng, uint32_t nPts, uint32_t nDim,
                            uint32_t nVec) {
  __shared__ dataPoint partial_sum[4 * 4 * 4 * BlockSizeWarp3D];
  register uint32_t f1, f2, f3;
  dataPoint d;
  dataPoint v1[4], v2[4], v3[4];
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
        d = y[ib[boxIdx] + TID] - (dataPoint)f1;

        v1[0] = g2(1 + d);
        v1[1] = g1(d);
        v1[2] = g1(1 - d);
        v1[3] = g2(2 - d);

        f2 = (uint32_t)floor(y[ib[boxIdx] + TID + nPts]);
        d = y[ib[boxIdx] + TID + nPts] - (dataPoint)f2;

        v2[0] = g2(1 + d);
        v2[1] = g1(d);
        v2[2] = g1(1 - d);
        v2[3] = g2(2 - d);

        f3 = (uint32_t)floor(y[ib[boxIdx] + TID + 2 * nPts]);
        d = y[ib[boxIdx] + TID + 2 * nPts] - (dataPoint)f3;
        v3[0] = g2(1 + d);
        v3[1] = g1(d);
        v3[2] = g1(1 - d);
        v3[3] = g2(2 - d);

        for (idx3 = 0; idx3 < 4; idx3++) {

          for (idx2 = 0; idx2 < 4; idx2++) {
            dataPoint qv =
                q[nPts * iVec + ib[boxIdx] + TID] * v2[idx2] * v3[idx3];
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
template void s2g(float *VGrid, float *y, float *VScat, uint32_t nGridDim,
                  uint32_t n, uint32_t d, uint32_t m, int *ib);
template void g2s(float *PhiScat, float *PhiGrid, float *y, uint32_t nGridDim,
                  uint32_t n, uint32_t d, uint32_t m);

template void s2g(double *VGrid, double *y, double *VScat, uint32_t nGridDim,
                  uint32_t n, uint32_t d, uint32_t m, int *ib);
template void g2s(double *PhiScat, double *PhiGrid, double *y,
                  uint32_t nGridDim, uint32_t n, uint32_t d, uint32_t m);
