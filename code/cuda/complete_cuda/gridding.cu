#include "gridding.cuh"
#include "matrix_indexing.hpp"
#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx4(i,j,k,l,m,n,o)  (SUB2IND4D(i,j,k,l,m,n,o))
__global__ void s2g1d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec, coord maxy) {
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
        atomicAdd(&V[f1 + idx1 + j * ng], qv * v1[idx1]);
        // V[f1 + idx1 + j * ng] += qv * v1[idx1]
      }
    }
  }
}
__global__ void s2g1drb(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
                uint32_t ng, uint32_t np, uint32_t nPts, uint32_t nDim,
                uint32_t nVec, coord maxy){
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    y[TID] /= maxy;
    if (y[TID] == 1) {
      y[TID] = y[TID] - 0.00000000000001;
    }
    y[TID] *= (ng - 3);
  }
  for (uint32_t s = 0; s < 2; s++) { // red-black sync

    for (uint32_t idual = 6*blockIdx.x; idual < (ng - 3); idual += 6*gridDim.x) { // coarse-grid

      for (uint32_t ifine = 0; ifine < 3; ifine++) { // fine-grid

        // get index of current grid box
        uint32_t i = 3 * s + idual + ifine;

        // if above boundaries, break
        if (i > ng - 4)
          break;

        // loop through all points inside box
        for (uint32_t TID = 0; TID < cb[i]; TID+=blockDim.x) {

          uint32_t f1;
          coord d;
          coord v1[4];

          f1 = (uint32_t)floor(y[ib[i]+TID]);
          d = y[ib[i]+TID] - (coord)f1;

          v1[0] = g2(1 + d);
          v1[1] = g1(d);
          v1[2] = g1(1 - d);
          v1[3] = g2(2 - d);

          for (uint32_t j = 0; j < nVec; j++) {

            coord qv = q[nPts * j +ib[i]+ TID];

            for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
              atomicAdd(&V[f1 + idx1 + j * ng], qv * v1[idx1]);


            } // (idx1)

          } // (j)

        } // (k)

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
                      uint32_t nDim, uint32_t nVec, coord maxy) {
  coord v1[4];
  coord v2[4];
  uint32_t f1;
  uint32_t f2;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    y[TID] /= maxy;
    if (y[TID] == 1) {
      y[TID] = y[TID] - 0.00000000000001;
    }
    y[TID] *= (ng - 3);

    y[TID + nPts] /= maxy;
    if (y[TID + nPts] == 1) {
      y[TID + nPts] = y[TID + nPts] - 0.00000000000001;
    }
    y[TID + nPts] *= (ng - 3);

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

__global__ void s2g3d(coord *V, coord *y, coord *q, uint32_t ng, uint32_t nPts,
                      uint32_t nDim, uint32_t nVec, coord maxy) {
  coord v1[4];
  coord v2[4];
  coord v3[4];
  uint32_t f1;
  uint32_t f2;
  uint32_t f3;
  coord d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    y[TID] /= maxy;
    if (y[TID] == 1) {
      y[TID] = y[TID] - 0.00000000000001;
    }
    y[TID] *= (ng - 3);

    y[TID + nPts] /= maxy;
    if (y[TID + nPts] == 1) {
      y[TID + nPts] = y[TID + nPts] - 0.00000000000001;
    }
    y[TID + nPts] *= (ng - 3);

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
