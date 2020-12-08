#include "gridding.hpp"
#include "common.hpp"
#include "matrix_indexing.hpp"
#include <cmath>
#include <iostream>
#include <string>
#define LAGRANGE_INTERPOLATION

#define y(i, j) y[SUB2IND2D((i), (j), nDim)]
#define q(i, j) q[SUB2IND2D((i), (j), nVec)]
#define Phi(i, j) Phi[SUB2IND2D((i), (j), nVec)]

#define V1(i, j, k) V[SUB2IND3D((i), (j), (k), ng, nVec)]
#define V2(i, j, k, l) V[SUB2IND4D((i), (j), (k), (l), ng, ng, nVec)]
#define V3(i, j, k, l, m)                                                      \
  V[SUB2IND5D((i), (j), (k), (l), (m), ng, ng, ng, nVec)]

void s2g1dCpu(coord *V, coord *y, coord *q, uint32_t ng, uint32_t np,
              uint32_t nPts, uint32_t nDim, uint32_t nVec) {

  uint32_t fprev=-1;
  for (uint32_t pid = 0; pid < np; pid++) {

    coord v1[4];

    for (uint32_t i = pid; i < nPts; i += np) {

      uint32_t f1;
      coord d;

      f1 = (uint32_t)floor(y(0, i));
      if(f1!=fprev){
        //printf("new f1=%d\n",f1 );
        if(f1<fprev+1){
          printf("---------Order Error------\n" );
        }
        fprev=f1;
      }
      d = y(0, i) - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);

      for (uint32_t j = 0; j < nVec; j++) {

        coord qv = q(j, i);

        for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
          V1(f1 + idx1, j, pid) += qv * v1[idx1];
        }
      }

    } // (i)

  } // (pid)
}

void s2g1drbCpu(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
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
          coord d;
          coord v1[4];

          f1 = (uint32_t)floor(y(0, ib[i] + k));
          if(i!=f1){
            //printf("f1=%d vs i=%d\n",f1,i );

          }

          d = y(0, ib[i] + k) - (coord)f1;

          v1[0] = g2(1 + d);
          v1[1] = g1(d);
          v1[2] = g1(1 - d);
          v1[3] = g2(2 - d);

          for (uint32_t j = 0; j < nVec; j++) {

            coord qv = q(j, ib[i] + k);

            for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
              V1(f1 + idx1, j, 0) += qv * v1[idx1];

            } // (idx1)

          } // (j)

        } // (k)

      } // (ifine)

    } // (idual)

  } // (s)

}

void s2g2dCpu(coord *V, coord *y, coord *q, uint32_t ng, uint32_t np,
              uint32_t nPts, uint32_t nDim, uint32_t nVec) {

  for (uint32_t pid = 0; pid < np; pid++) {

    coord v1[4];
    coord v2[4];

    for (uint32_t i = pid; i < nPts; i += np) {

      uint32_t f1, f2;
      coord d;

      f1 = (uint32_t)floor(y(0, i));
      d = y(0, i) - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);

      f2 = (uint32_t)floor(y(1, i));
      d = y(1, i) - (coord)f2;

      v2[0] = g2(1 + d);
      v2[1] = g1(d);
      v2[2] = g1(1 - d);
      v2[3] = g2(2 - d);
      printf("i=%d f1 =%d ,f2=%d  \n",i,f1 ,f2  );

      for (uint32_t j = 0; j < nVec; j++) {

        for (uint32_t idx2 = 0; idx2 < 4; idx2++) {

          coord qv2 = q(j, i) * v2[idx2];

          for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
            V2(f1 + idx1, f2 + idx2, j, pid) += qv2 * v1[idx1];
          }
        }
      }

    } // (i)

  } // (pid)
}

void s2g2drbCpu(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
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

          uint32_t f1, f2;
          coord d;
          coord v1[4], v2[4];

          f1 = (uint32_t)floor(y(0, ib[i] + k));
          d = y(0, ib[i] + k) - (coord)f1;

          v1[0] = g2(1 + d);
          v1[1] = g1(d);
          v1[2] = g1(1 - d);
          v1[3] = g2(2 - d);

          f2 = (uint32_t)floor(y(1, ib[i] + k));
          d = y(1, ib[i] + k) - (coord)f2;

          v2[0] = g2(1 + d);
          v2[1] = g1(d);
          v2[2] = g1(1 - d);
          v2[3] = g2(2 - d);
          //if(i==4){printf(" f1=%d f2=%d\n",f1,f2   );}
          for (uint32_t j = 0; j < nVec; j++) {

            for (uint32_t idx2 = 0; idx2 < 4; idx2++) {

              coord qv2 = q(j, ib[i] + k) * v2[idx2];

              for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
                V2(f1 + idx1, f2 + idx2, j, 0) += qv2 * v1[idx1];

              } // (idx1)

            } // (idx2)

          } // (j)

        } // (k)

      } // (ifine)

    } // (idual)

  } // (s)
}

void s2g3dCpu(coord *V, coord *y, coord *q, uint32_t ng, uint32_t np,
              uint32_t nPts, uint32_t nDim, uint32_t nVec) {

  for (uint32_t pid = 0; pid < np; pid++) {

    coord v1[4];
    coord v2[4];
    coord v3[4];

    for (uint32_t i = pid; i < nPts; i += np) {

      uint32_t f1, f2, f3;
      coord d;

      f1 = (uint32_t)floor(y(0, i));
      d = y(0, i) - (coord)f1;

      v1[0] = g2(1 + d);
      v1[1] = g1(d);
      v1[2] = g1(1 - d);
      v1[3] = g2(2 - d);

      f2 = (uint32_t)floor(y(1, i));
      d = y(1, i) - (coord)f2;

      v2[0] = g2(1 + d);
      v2[1] = g1(d);
      v2[2] = g1(1 - d);
      v2[3] = g2(2 - d);

      f3 = (uint32_t)floor(y(2, i));
      d = y(2, i) - (coord)f3;

      v3[0] = g2(1 + d);
      v3[1] = g1(d);
      v3[2] = g1(1 - d);
      v3[3] = g2(2 - d);

      for (uint32_t j = 0; j < nVec; j++) {

        for (uint32_t idx3 = 0; idx3 < 4; idx3++) {

          for (uint32_t idx2 = 0; idx2 < 4; idx2++) {

            coord qv2v3 = q(j, i) * v2[idx2] * v3[idx3];

            for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
              V3(f1 + idx1, f2 + idx2, f3 + idx3, j, pid) += qv2v3 * v1[idx1];
            }
          }
        }
      }

    } // (i)

  } // (pid)
}

void s2g3drbCpu(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
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

          uint32_t f1, f2, f3;
          coord d;
          coord v1[4], v2[4], v3[4];

          f1 = (uint32_t)floor(y(0, ib[i] + k));
          d = y(0, ib[i] + k) - (coord)f1;

          v1[0] = g2(1 + d);
          v1[1] = g1(d);
          v1[2] = g1(1 - d);
          v1[3] = g2(2 - d);

          f2 = (uint32_t)floor(y(1, ib[i] + k));
          d = y(1, ib[i] + k) - (coord)f2;

          v2[0] = g2(1 + d);
          v2[1] = g1(d);
          v2[2] = g1(1 - d);
          v2[3] = g2(2 - d);

          f3 = (uint32_t)floor(y(2, ib[i] + k));
          d = y(2, ib[i] + k) - (coord)f3;

          v3[0] = g2(1 + d);
          v3[1] = g1(d);
          v3[2] = g1(1 - d);
          v3[3] = g2(2 - d);

          for (uint32_t j = 0; j < nVec; j++) {

            for (uint32_t idx3 = 0; idx3 < 4; idx3++) {

              for (uint32_t idx2 = 0; idx2 < 4; idx2++) {

                coord qv2v3 = q(j, ib[i] + k) * v2[idx2] * v3[idx3];

                for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
                  V3(f1 + idx1, f2 + idx2, f3 + idx3, j, 0) += qv2v3 * v1[idx1];

                } // (idx1)

              } // (idx2)

            } // (idx3)

          } // (j)

        } // (k)

      } // (ifine)

    } // (idual)

  } // (s)
}

void g2s1dCpu(coord *Phi, coord *V, coord *y, uint32_t ng, uint32_t nPts,
              uint32_t nDim, uint32_t nVec) {

  for (uint32_t i = 0; i < nPts; i++) {

    uint32_t f1;
    coord d;

    coord v1[4];

    f1 = (uint32_t)floor(y(0, i));
    d = y(0, i) - (coord)f1;

    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {

      coord accum = 0;

      for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
        accum += V1(f1 + idx1, j, 0) * v1[idx1];
      }

      Phi(j, i) = accum;
    }

  } // (i)
}

void g2s2dCpu(coord *Phi, coord *V, coord *y, uint32_t ng, uint32_t nPts,
              uint32_t nDim, uint32_t nVec) {

  for (uint32_t i = 0; i < nPts; i++) {

    uint32_t f1, f2;
    coord d;

    coord v1[4];
    coord v2[4];

    f1 = (uint32_t)floor(y(0, i));
    d = y(0, i) - (coord)f1;

    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y(1, i));
    d = y(1, i) - (coord)f2;

    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {

      coord accum = 0;

      for (uint32_t idx2 = 0; idx2 < 4; idx2++) {
        coord qv2 = v2[idx2];

        for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
          accum += V2(f1 + idx1, f2 + idx2, j, 0) * qv2 * v1[idx1];
        }
      }

      Phi(j, i) = accum;
    }

  } // (i)
}

void g2s3dCpu(coord *Phi, coord *V, coord *y, uint32_t ng, uint32_t nPts,
              uint32_t nDim, uint32_t nVec) {

  for (uint32_t i = 0; i < nPts; i++) {

    uint32_t f1, f2, f3;
    coord d;

    coord v1[4];
    coord v2[4];
    coord v3[4];

    f1 = (uint32_t)floor(y(0, i));
    d = y(0, i) - (coord)f1;

    v1[0] = g2(1 + d);
    v1[1] = g1(d);
    v1[2] = g1(1 - d);
    v1[3] = g2(2 - d);

    f2 = (uint32_t)floor(y(1, i));
    d = y(1, i) - (coord)f2;

    v2[0] = g2(1 + d);
    v2[1] = g1(d);
    v2[2] = g1(1 - d);
    v2[3] = g2(2 - d);

    f3 = (uint32_t)floor(y(2, i));
    d = y(2, i) - (coord)f3;

    v3[0] = g2(1 + d);
    v3[1] = g1(d);
    v3[2] = g1(1 - d);
    v3[3] = g2(2 - d);

    for (uint32_t j = 0; j < nVec; j++) {

      coord accum = 0;

      for (uint32_t idx3 = 0; idx3 < 4; idx3++) {

        for (uint32_t idx2 = 0; idx2 < 4; idx2++) {

          coord qv2v3 = v2[idx2] * v3[idx3];

          for (uint32_t idx1 = 0; idx1 < 4; idx1++) {
            accum +=
                V3(f1 + idx1, f2 + idx2, f3 + idx3, j, 0) * qv2v3 * v1[idx1];
          }
        }
      }

      Phi(j, i) = accum;
    }

  } // (i)
}
