#include "gridding.cuh"
#include "matrix_indexing.hpp"
#include "utils_gpu.cuh"
#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx4(i, j, k, l, m, n, o) (SUB2IND4D(i, j, k, l, m, n, o))
void s2g(coord *V, coord *y, coord *q, uint32_t ng, uint32_t n,
                      uint32_t d, uint32_t nVec){

                          switch (d) {

                            case 1:
                              s2g1d<<<64, 512>>>(V, y, q, ng , n, d, nVec);
                              break;
                            case 2:
                              s2g2d<<<64, 512>>>(V, y, q, ng , n, d, nVec);
                              break;
                            case 3:
                              s2g3d<<<64, 512>>>(V, y, q, ng , n, d, nVec);
                              break;
                          }


}
void s2grb(coord *V, coord *y, coord *q,uint32_t* ib,uint32_t* cb, uint32_t ng, uint32_t n,
                      uint32_t d, uint32_t nVec){

                          switch (d) {

                            case 1:
                              s2g1drb<<<64, 32>>>(V, y, q,ib,cb, ng , n, d, nVec);
                              break;
                            case 2:
                              s2g2drb<<<64, 512>>>(V, y, q,ib,cb, ng , n, d, nVec);
                              break;
                            case 3:
                              s2g3drb<<<64, 512>>>(V, y, q,ib,cb, ng , n, d, nVec);
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
#define BlockSize 32
#define warpsize 32
__global__ void s2g1drbwarp(coord *V, coord *y, coord *q, uint32_t *ib,
                         uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec) {
                          __shared__ coord partial_sum[4 * 2 * BlockSize];
                          register uint32_t f1;
                          coord d;
                          coord v1[4];
                          register uint32_t i;
                          register uint32_t idx1;
                          register uint32_t tid = blockIdx.x*blockDim.x+threadIdx.x;
                          uint32_t lane = tid % 32;
                          uint32_t warpid=(uint32_t)tid/32;
                          for (i=warp_id;i<ng-3;i+=gridDim.x*blockDim.x/32){
                            // loop through all points inside box
                            for (int x = 0; x < 8; x++) {
                              partial_sum[x * BlockSize + tid] = 0;
                            }
                            for (uint32_t TID = lane; TID < ib[i + 1] - ib[i];TID += warpsize) {

                              f1 = (uint32_t)floor(y[ib[i] + TID]);
                              d = y[ib[i] + TID] - (coord)f1;

                              v1[0] = g2(1 + d);
                              v1[1] = g1(d);
                              v1[2] = g1(1 - d);
                              v1[3] = g2(2 - d);
                              for (uint32_t j = 0; j < nVec; j++) {

                                coord qv = q[nPts * j + ib[i] + TID];

                                for (idx1 = 0; idx1 < 4; idx1++) {
                                  partial_sum[(idx1 * nVec + j) * BlockSize + threadIdx.x] += qv * v1[idx1];
                                } // (idx1)

                              } // (j)

                            } // (k)
                            coord val;
                            for (uint32_t j = 0; j < nVec; j++) {
                              for (idx1 = 0; idx1 < 4; idx1++) {
                                val =
                                    warp_reduce(partial_sum[(idx1 * nVec + j) * BlockSize + threadIdx.x]);
                                    if (lane == 0) {
                                      atomicAdd(&V[i + idx1 + j * ng], val);}
                              }}

                          }

}
__global__ void s2g1drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec) {
  __shared__ coord partial_sum[4 * 2 * BlockSize];
  register uint32_t f1;
  coord d;
  coord v1[4];
  register uint32_t i;
  register uint32_t idx1;
  register uint32_t tid = threadIdx.x;
  uint32_t lane = tid % 32;
  uint32_t warpid=(uint32_t)tid/32;
  for (uint32_t s = 0; s < 2; s++) { // red-black sync

    for (uint32_t idual = 6 * blockIdx.x; idual < (ng - 3);
         idual += 6 * gridDim.x) { // coarse-grid

      for (uint32_t ifine = 0; ifine < 3; ifine++) { // fine-grid

        // get index of current grid box
        i = 3 * s + idual + ifine;

        // if above boundaries, break
        if (i > ng - 4)
          break;

        // loop through all points inside box
        for (int x = 0; x < 8; x++) {
          partial_sum[x * BlockSize + tid] = 0;
        }

        for (uint32_t TID = threadIdx.x; TID < ib[i + 1] - ib[i];TID += blockDim.x) {

          f1 = (uint32_t)floor(y[ib[i] + TID]);
          d = y[ib[i] + TID] - (coord)f1;

          v1[0] = g2(1 + d);
          v1[1] = g1(d);
          v1[2] = g1(1 - d);
          v1[3] = g2(2 - d);

          for (uint32_t j = 0; j < nVec; j++) {

            coord qv = q[nPts * j + ib[i] + TID];

            for (idx1 = 0; idx1 < 4; idx1++) {
              partial_sum[(idx1 * nVec + j) * BlockSize + tid] += qv * v1[idx1];
            } // (idx1)

          } // (j)

        } // (k)
        coord val;
        for (uint32_t j = 0; j < nVec; j++) {
          for (idx1 = 0; idx1 < 4; idx1++) {

            for (uint32_t s = blockDim.x / 2; s > 32; s >>= 1) {
              if (tid < s) {
                partial_sum[(idx1 * nVec + j) * BlockSize + tid] +=
                    partial_sum[(idx1 * nVec + j) * BlockSize + tid + s];
              }
              __syncthreads();
            }
            if (tid < 32) {
              val =
                  warp_reduce(partial_sum[(idx1 * nVec + j) * BlockSize + tid]);
            }
            if (lane == 0) {
              atomicAdd(&V[i + idx1 + j * ng], val);
              // V[i + idx1 + j * ng]+=val;
            }
          }
        }

      } // (ifine)

    } // (idual)

  } // (s)
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
#define BlockSizeD2 32

__inline__ __device__ coord Reduce2V2D(uint32_t idx1, uint32_t idx2,
                                       coord *partial_sum, coord *V,
                                       uint32_t f1, uint32_t f2, uint32_t tid,
                                       uint32_t nVec, uint32_t ng) {
  coord val;
  for (uint32_t j = 0; j < nVec; j++) {
    for (idx1 = 0; idx1 < 4; idx1++) {
      for (idx2 = 0; idx2 < 4; idx2++) {

        if (tid < 32) {
          val = warp_reduce(
              partial_sum[(idx2 * nVec * 4 + idx1 * nVec + j) * BlockSizeD2 +
                          tid]);
        }
        if (tid == 0) {
          atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng], val);
          // V[i + idx1 + j * ng]+=val;
        }
      }
    }
  }

  return val;
}
__global__ void s2g2drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec) {
  __shared__ coord partial_sum[4 * 4 * 3 * BlockSizeD2];
  register uint32_t f1, f2;
  coord d;
  coord v1[4], v2[4];
  register uint32_t i;
  register uint32_t idx1, idx2;
  register uint32_t tid = threadIdx.x % 32;
  for (uint32_t s = 0; s < 2; s++) { // red-black sync

    for (uint32_t idual = 6 * blockIdx.x; idual < (ng - 3);
         idual += 6 * gridDim.x) { // coarse-grid

      for (uint32_t ifine = 0; ifine < 3; ifine++) { // fine-grid

        // get index of current grid box
        i = 3 * s + idual + ifine;

        // if above boundaries, break
        if (i > ng - 2)
          {printf("do you\n" );
          break;}


        for (uint32_t x = 0; x < ng - 4; x++) {
          uint32_t box = i * (ng - 4) + x;
          for (int temp = 0; temp < 4 * 4 * 3; temp++) {
           partial_sum[temp * BlockSizeD2 + tid] = 0;
          }
          // loop through all points inside box
          for (uint32_t TID = threadIdx.x; TID < ib[box + 1] - ib[box];
               TID += blockDim.x) {

            f1 = (uint32_t)floor(y[ib[box] + TID]);
            d = y[ib[box] + TID] - (coord)f1;

            v1[0] = g2(1 + d);
            v1[1] = g1(d);
            v1[2] = g1(1 - d);
            v1[3] = g2(2 - d);

            f2 = (uint32_t)floor(y[ib[box] + TID + nPts]);
            d = y[ib[box] + TID + nPts] - (coord)f2;

            v2[0] = g2(1 + d);
            v2[1] = g1(d);
            v2[2] = g1(1 - d);
            v2[3] = g2(2 - d);

            printf("f1=%d f2=%d i=%d j=%d box=%d\n", f1, f2, i, x,box);

            for (uint32_t j = 0; j < nVec; j++) {
              for (idx2 = 0; idx2 < 4; idx2++) {

                coord qv = q[nPts * j + ib[box] + TID] * v2[idx2];

                for (idx1 = 0; idx1 < 4; idx1++) {
                  //atomicAdd(&V[f1 + idx1 + (f2 + idx2) * ng + j * ng * ng], qv * v1[idx1]);

                  partial_sum[(idx2 * nVec * 4 + idx1 * nVec + j) *
                                  BlockSizeD2 +
                              tid] += qv * v1[idx1];
                   //if(i==0){printf(" partial_sum =%lf\n",partial_sum[(idx2 *nVec * 4 + idx1 * nVec + j) * 32 + tid] );}
                  // idx4(tid, j, idx1, idx2, 32, nVec, 4)

                } // (idx1)

              } // (j)

            } // (k)



          } // (ifine)
          Reduce2V2D(idx1, idx2, partial_sum, V, f1, f2, tid, nVec, ng);
        }

      } // (idual)

    } // (s)
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

__global__ void s2g3drb(coord *V, coord *y, coord *q, uint32_t *ib,
                        uint32_t *cb, uint32_t ng, uint32_t nPts, uint32_t nDim,
                        uint32_t nVec) {
  coord v1[4];
  coord v2[4];
  coord v3[4];
  uint32_t f1;
  uint32_t f2;
  uint32_t f3;
  coord d;

  for (uint32_t s = 0; s < 2; s++) { // red-black sync

    for (uint32_t idual = 6 * blockIdx.x; idual < (ng - 3);
         idual += 6 * gridDim.x) { // coarse-grid

      for (uint32_t ifine = 0; ifine < 3; ifine++) { // fine-grid

        // get index of current grid box
        uint32_t i = 3 * s + idual + ifine;

        // if above boundaries, break
        if (i > ng - 4)
          break;
        for (uint32_t j = 0; j < ng - 2; j++) {
          uint32_t box = i * ng + j;
          // loop through all points inside box
          for (uint32_t TID = 0; TID < ib[i * +1] - ib[i]; TID += blockDim.x) {

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
                    atomicAdd(&V[idx4(f1 + idx1, f2 + idx2, f3 + idx3, j, ng,
                                      ng, ng)],
                              qv * v1[idx1]);
                  }
                }
              }
            }

          } // (k)
        }

      } // (ifine)

    } // (idual)

  } // (s)
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
