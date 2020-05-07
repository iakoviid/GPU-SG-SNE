#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define LAGRANGE_INTERPOLATION

#ifdef LAGRANGE_INTERPOLATION

__inline__ __host__ __device__ double g1(double d) {
  return 0.5 * d * d * d - 1.0 * d * d - 0.5 * d + 1;
}

__inline__ __host__ __device__ double g2(double d) {
  double cc = 1.0 / 6.0;
  return -cc * d * d * d + 1.0 * d * d - 11 * cc * d + 1;
}

#else

__inline__ __host__ __device__ double g1(double d) {
  return 1.5 * d * d * d - 2.5 * d * d + 1;
}

__inline__ __host__ __device__ double g2(double d) {
  return -0.5 * d * d * d + 2.5 * d * d - 4 * d + 2;
}

#endif

#define SUB2IND2DVECSTART(j, m) ((m) * ((j)))
#define SUB2IND3DVECSTART(j, k, m, n) ((m) * ((n) * (k) + (j)))
#define SUB2IND4DVECSTART(j, k, l, m, n, o)                                    \
  ((m) * ((n) * ((o) * (l) + (k)) + (j)))
#define SUB2IND5DVECSTART(j, k, l, a, m, n, o, p)                              \
  ((m) * ((n) * ((o) * ((p) * (a) + (l)) + (k)) + (j)))

#define SUB2IND2D(i, j, m) (SUB2IND2DVECSTART(j, m) + (i))
#define SUB2IND3D(i, j, k, m, n) (SUB2IND3DVECSTART(j, k, m, n) + (i))
#define SUB2IND4D(i, j, k, l, m, n, o)                                         \
  (SUB2IND4DVECSTART(j, k, l, m, n, o) + (i))
#define SUB2IND5D(i, j, k, l, a, m, n, o, p)                                   \
  (SUB2IND5DVECSTART(j, k, l, a, m, n, o, p) + (i))

#define y(i, j) y[SUB2IND2D((i), (j), nDim)]
#define q(i, j) q[SUB2IND2D((i), (j), nVec)]
#define Phi(i, j) Phi[SUB2IND2D((i), (j), nVec)]

#define V1(i, j, k) V[SUB2IND3D((i), (j), (k), ng, nVec)]
#define V2(i, j, k, l) V[SUB2IND4D((i), (j), (k), (l), ng, ng, nVec)]
#define V3(i, j, k, l, m)                                                      \
  V[SUB2IND5D((i), (j), (k), (l), (m), ng, ng, ng, nVec)]

__global__ void s2g1dCuda(double *V, double *y, double *q, uint32_t ng,
                          uint32_t nPts, uint32_t nDim, uint32_t nVec,
                          double maxy) {
  double v1[4];
  uint32_t f1;
  double d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    y[TID] /= maxy;
    if (y[TID] == 1) {
      y[TID] = y[TID] - 0.00000000000001;
    }
    y[TID] *= (ng - 3);
    // printf("ycuda[%d]=%lf \n",TID,y[TID] );
    f1 = (uint32_t)floor(y[TID]);
    // printf(" TID=%d f1=%d\n",TID  ,f1 );
    d = y[TID] - (double)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (int j = 0; j < nVec; j++) {
      double qv = q[nPts * j + TID];
      // printf("CUDA q[%d,%d]=%lf\n",TID,j,qv );
      for (int idx1 = 0; idx1 < 4; idx1++) {
        V[f1 + idx1 + j * ng] += qv * v1[idx1];
      }
    }
  }
}

void s2g1d(double *V, double *y, double *q, uint32_t ng, uint32_t np,
           uint32_t nPts, uint32_t nDim, uint32_t nVec) {

  double v1[4];
  uint32_t prevf1 = 0;
  for (uint32_t i = 0; i < nPts; i++) {

    uint32_t f1;
    double d;
    // printf("y[%d]=%lf\n",i,y[i] );
    f1 = (uint32_t)floor(y(0, i));

    if (prevf1 != f1) {
      printf("host prev=%d vs f1=%d in i=%d \n", prevf1, f1, i);
    }
    prevf1 = f1;

    d = y(0, i) - (double)f1;

    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {

      double qv = q(j, i);
      // printf("host q[%d,%d]=%lf ===== vs ====
      // q[%d,%d]=%lf,\n",i,j,qv,i,j,q[i*nVec+j] );
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        V1(f1 + idx1, j, 0) += qv * v1[idx1];
      }
    }

  } // (i)
}

void s2g1drb(double *V, double *y, double *q, uint32_t *ib, uint32_t *cb,
             uint32_t ng, uint32_t np, uint32_t nPts, uint32_t nDim,
             uint32_t nVec) {

  for (uint32_t s = 0; s < 2; s++) { // red-black sync

    for (uint32_t idual = 0; idual < (ng - 3); idual += 6) { // coarse-grid

      for (uint32_t ifine = 0; ifine < 3; ifine++) { // fine-grid

        // get index of current grid box
        uint32_t i = 3 * s + idual + ifine;

        // if above boundaries, break
        if (i > ng - 4)
          break;

        // loop through all points inside box
        for (uint32_t k = 0; k < cb[i]; k++) {

          uint32_t f1;
          double d;
          double v1[4];

          f1 = (uint32_t)floor(y(0, ib[i] + k));

          d = y(0, ib[i] + k) - (double)f1;

          v1[0] = g2(1 + d);
          v1[1] = g1(d);
          v1[2] = g1(1 - d);
          v1[3] = g2(2 - d);

          for (uint32_t j = 0; j < nVec; j++) {

            double qv = q(j, ib[i] + k);

            for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
              V1(f1 + idx1, j, 0) += qv * v1[idx1];

            } // (idx1)

          } // (j)

        } // (k)

      } // (ifine)

    } // (idual)

  } // (s)
}

__global__ void g2s1dCuda(double *Phi, double *V, double *y, uint32_t ng,
                          uint32_t nPts, uint32_t nDim, uint32_t nVec) {
  double v1[4];
  uint32_t f1;
  double d;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    f1 = (uint32_t)floor(y[TID]);
    d = y[TID] - (double)f1;
    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {
      double accum = 0;
      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        printf("CUDA, V[%d]=%lf\n",f1 + idx1 + j * ng,V[f1 + idx1 + j * ng]  );
        accum += V[f1 + idx1 + j * ng] * v1[idx1];
      }
      Phi[TID + j * nPts] = accum;
    }
  }
}

void g2s1d(double *Phi, double *V, double *y, uint32_t ng, uint32_t nPts,
           uint32_t nDim, uint32_t nVec) {

  for (uint32_t i = 0; i < nPts; i++) {

    uint32_t f1;
    double d;

    double v1[4];

    f1 = (uint32_t)floor(y(0, i));
    d = y(0, i) - (double)f1;

    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {

      double accum = 0;

      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        //printf("Host, V[%d]=%lf\n",i,V1(f1 + idx1, j, 0)  );

        accum += V1(f1 + idx1, j, 0) * v1[idx1];
      }

      Phi(j, i) = accum;
    }

  } // (i)
}
