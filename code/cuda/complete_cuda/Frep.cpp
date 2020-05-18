#include "common.hpp"
#include <cmath>
#include <limits>
#include "utils.cuh"
#include "relocateData.hpp"
#include "nuconv.hpp"

coord computeFrepulsive_exact(coord *frep, coord *pointsX, int N, int d) {

  coord *zetaVec = (coord *)calloc(N, sizeof(coord));

  for (int i = 0; i < N; i++) {
    coord Yi[10] = {0};
    for (int dd = 0; dd < d; dd++)
      Yi[dd] = pointsX[i * d + dd];

    coord Yj[10] = {0};

    for (int j = 0; j < N; j++) {

      if (i != j) {

        coord dist = 0.0;
        for (int dd = 0; dd < d; dd++) {
          Yj[dd] = pointsX[j * d + dd];
          dist += (Yj[dd] - Yi[dd]) * (Yj[dd] - Yi[dd]);
        }

        for (int dd = 0; dd < d; dd++) {
          frep[i * d + dd] += (Yi[dd] - Yj[dd]) / ((1 + dist) * (1 + dist));
        }

        zetaVec[i] += 1.0 / (1.0 + dist);
      }
    }
  }
  coord zeta = 0;
  for (int i = 0; i < N; i++) {
    zeta = zeta + zetaVec[i];
  }

  for (int i = 0; i < N; i++) {
    for (int j = 0; j < d; j++) {
      frep[(i * d) + j] /= zeta;
    }
  }

  free(zetaVec);

  return zeta;
}

template <typename dataval>
dataval zetaAndForce(dataval *const F,            // Forces
                     const dataval *const Y,      // Coordinates
                     const dataval *const Phi,    // Values
                     const uint32_t *const iPerm, // Permutation
                     const uint32_t nPts,         // #points
                     const uint32_t nDim) {       // #dimensions

  dataval Z = 0;

  // compute normalization term
  for (uint32_t i = 0; i < nPts; i++) {
    dataval Ysq = 0;
    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[i * nDim + j] * Y[i * nDim + j];
      Z -= 2 * (Y[i * nDim + j] * Phi[i * (nDim + 1) + j + 1]);
    }
    Z += (1 + 2 * Ysq) * Phi[i * (nDim + 1)];
  }

  Z = Z - nPts;

  // Compute repulsive forces
  for (uint32_t i = 0; i < nPts; i++) {
    for (uint32_t j = 0; j < nDim; j++)
      F[iPerm[i] * nDim + j] = (Y[i * nDim + j] * Phi[i * (nDim + 1)] -
                                Phi[i * (nDim + 1) + j + 1]) /
                               Z;
  }

  return Z;
}


coord computeFrepulsive_interpCPU(coord *Frep, coord *y, int n, int d, double h,
                               int np) {

  // ~~~~~~~~~~ make temporary data copies
  coord *yt = static_cast<coord *>(malloc(n * d * sizeof(coord)));
  coord *yr = static_cast<coord *>(malloc(n * d * sizeof(coord)));

  // struct timeval start;

  // ~~~~~~~~~~ move data to (0,0,...)
  coord miny[d];
  for (int i = 0; i < d; i++) {
    miny[i] = std::numeric_limits<coord>::infinity();
  }

  //--G cauch is translationaly invariant
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++)
      miny[j] = miny[j] > y[i * d + j] ? y[i * d + j] : miny[j];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      y[i * d + j] -= miny[j];
    }
  }

  // ~~~~~~~~~~ find maximum value (across all dimensions) and get grid size
  //--G I have something similar max(maxy/h,14) vs max((maxy-miny)*2,20)
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;

  int nGrid = std::max((int)std::ceil(maxy / h), 14);
  nGrid = getBestGridSize(nGrid);

  //#ifdef VERBOSE
  std::cout << "Grid: " << nGrid << " h: " << h << "maxy: " << maxy
            << std::endl;
  //#endif

  // ~~~~~~~~~~ setup inputs to nuConv

  std::copy(y, y + (n * d), yt);

  coord *VScat = (coord *)malloc(n * (d + 1) * sizeof(coord));
  coord *PhiScat = (coord *)calloc(n * (d + 1), sizeof(coord));
  uint32_t *iPerm = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *ib = (uint32_t *)calloc(nGrid, sizeof(uint32_t));
  uint32_t *cb = (uint32_t *)calloc(nGrid, sizeof(uint32_t));

  for (int i = 0; i < n; i++) {
    iPerm[i] = i;
  }

  // start = tsne_start_timer();
  //relocateCoarseGridCPU(&yt, &iPerm, ib, cb, n, nGrid, d, np);
  relocateCoarseGridCPU(&yt,&iPerm,ib,cb,n,nGrid,d,np);


  /*
  if (timeInfo != nullptr)
    timeInfo[0] = tsne_stop_timer("Gridding", start);
  else
    tsne_stop_timer("Gridding", start);
    */
  // ----- setup VScat (value on scattered points)

  for (int i = 0; i < n; i++) {

    VScat[i * (d + 1)] = 1.0;
    for (int j = 0; j < d; j++)
      VScat[i * (d + 1) + j + 1] = yt[i * d + j];
  }

  std::copy(yt, yt + (n * d), yr);
  nuconvCPU(PhiScat, yt, VScat, ib, cb, n, d, d + 1, np, nGrid);

  // ~~~~~~~~~~ run nuConv
  /*
  if (timeInfo != nullptr)
    nuconv(PhiScat, yt, VScat, ib, cb, n, d, d + 1, np, nGrid, &timeInfo[1]);
  else
    nuconv(PhiScat, yt, VScat, ib, cb, n, d, d + 1, np, nGrid);
  */
  // ~~~~~~~~~~ compute Z and repulsive forces

  // start = tsne_start_timer();
  coord zeta = zetaAndForce(Frep, yr, PhiScat, iPerm, n, d);
  /*
  if (timeInfo != NULL)
    timeInfo[4] = tsne_stop_timer("F&Z", start);
  else
    tsne_stop_timer("F&Z", start);
  */
  free(yt);
  free(yr);
  free(VScat);
  free(PhiScat);
  free(iPerm);
  free(ib);
  free(cb);
  return zeta;
}
