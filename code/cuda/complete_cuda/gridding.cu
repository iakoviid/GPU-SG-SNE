#include "gridding.cuh"


__global__ void s2g1d(coord *V, coord *y, coord *q, uint32_t ng,
                          uint32_t nPts, uint32_t nDim, uint32_t nVec,
                          coord maxy) {
  coord v1[4];
  uint32_t f1;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    y[TID] /= maxy;
    if (y[TID] == 1) {
      y[TID] = y[TID] - 0.00000000000001;
    }
    y[TID] *= (ng - 3);

    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (coord)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      coord qv = q[nPts * j + TID];
      for (int idx1 = 0; idx1 < 4; idx1++) {
        atomicAdd(&V[f1 + idx1 + j * ng],qv * v1[idx1]);
        //V[f1 + idx1 + j * ng] += qv * v1[idx1]
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
        //printf("CUDA, V[%d]=%lf\n",f1 + idx1 + j * ng,V[f1 + idx1 + j * ng]  );
        accum += V[f1 + idx1 + j * ng] * v1[idx1];
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}
