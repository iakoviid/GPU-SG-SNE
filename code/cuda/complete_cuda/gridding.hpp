#ifndef GRIDDING_HPP
#define GRIDDING_HPP

#include "gridding.cuh"
void s2g1dCpu(coord *V, coord *y, coord *q, uint32_t ng, uint32_t np,
              uint32_t nPts, uint32_t nDim, uint32_t nVec);
void g2s1dCpu(coord *Phi, coord *V, coord *y, uint32_t ng, uint32_t nPts,
              uint32_t nDim, uint32_t nVec);
void s2g2dCpu(coord *V, coord *y, coord *q, uint32_t ng, uint32_t np,
              uint32_t nPts, uint32_t nDim, uint32_t nVec);
void g2s2dCpu(coord *Phi, coord *V, coord *y, uint32_t ng, uint32_t nPts,
              uint32_t nDim, uint32_t nVec);
void s2g1drbCpu(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
                uint32_t ng, uint32_t np, uint32_t nPts, uint32_t nDim,
                uint32_t nVec);
void s2g2drbCpu(coord *V, coord *y, coord *q, uint32_t *ib, uint32_t *cb,
                uint32_t ng, uint32_t np, uint32_t nPts, uint32_t nDim,
                uint32_t nVec);
#endif
