
#include "nuconv.hpp"
#include "gridding.hpp"
#include "non_periodic_conv.hpp"
#include "utils.cuh"

void nuconvCPU(coord *PhiScat, coord *y, coord *VScat, uint32_t *ib,
               uint32_t *cb, int n, int d, int m, int np, int nGridDim,
               double *timeInfo) {
  struct timeval start;

  // ~~~~~~~~~~ normalize coordinates (inside bins)

  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for (int i = 0; i < n * d; i++) {
    y[i] /= maxy;

    // ~~~~~~~~~~ scale them from 0 to ng-1

    if (1 == y[i])
      y[i] = y[i] - std::numeric_limits<coord>::epsilon();

    y[i] *= (nGridDim - 1);
  }
  for (int i = 0; i < n * d; i++)
    if ((y[i] >= nGridDim - 1) || (y[i] < 0))
      exit(1);

  // ~~~~~~~~~~ find exact h

  double h = maxy / (nGridDim - 1 - std::numeric_limits<coord>::epsilon());

  // ~~~~~~~~~~ scat2grid
  int szV = pow(nGridDim + 2, d) * m;
  coord *VGrid = static_cast<coord *>(calloc(szV * np, sizeof(coord)));

  start = tsne_start_timer();

  switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD) {
      s2g1dCpu(VGrid, y, VScat, nGridDim + 2, np, n, d, m);
    } else
      s2g1dCpu(VGrid, y, VScat, nGridDim + 2, np, n, d, m);

    break;

  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD)

      s2g2dCpu(VGrid, y, VScat, nGridDim + 2, np, n, d, m);
    else
      s2g2dCpu(VGrid, y, VScat, nGridDim + 2, np, n, d, m);

    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD) {

      s2g3dCpu(VGrid, y, VScat, nGridDim + 2, np, n, d, m);
    } else {
      s2g3dCpu(VGrid, y, VScat, nGridDim + 2, np, n, d, m);
    }

    break;
  }

  if (timeInfo != NULL)
    timeInfo[0] = tsne_stop_timer("S2G", start);
  else
    tsne_stop_timer("S2G", start);

  int print=0;
  if(print==1){
  extractEmbeddingText(VGrid,pow(nGridDim + 2, d),m,"VGrid_cpu.txt");
  }

  // ~~~~~~~~~~ grid2grid
  coord *PhiGrid = static_cast<coord *>(calloc(szV, sizeof(coord)));
  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = nGridDim + 2;
  }
  start = tsne_start_timer();

  switch (d) {

  case 1:
    conv1dnopad(PhiGrid, VGrid, h, nGridDims, m, d, np);
    break;

  case 2:

    conv2dnopad(PhiGrid, VGrid, h, nGridDims, m, d, np);
    break;

  case 3:
    conv3dnopad(PhiGrid, VGrid, h, nGridDims, m, d, np);
    break;
  }
  if (timeInfo != NULL)
    timeInfo[1] = tsne_stop_timer("G2G", start);
  else
    tsne_stop_timer("G2G", start);

  if(print==1){
  extractEmbeddingText(PhiGrid,pow(nGridDim + 2, d),m,"PhiGrid_cpu.txt");
  }
  // ~~~~~~~~~~ grid2scat
  start = tsne_start_timer();

  switch (d) {

  case 1:
    g2s1dCpu(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;

  case 2:
    g2s2dCpu(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;

  case 3:
    g2s3dCpu(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;
  }

  if (timeInfo != NULL)
    timeInfo[2] = tsne_stop_timer("G2S", start);
  else
    tsne_stop_timer("G2S", start);

  // ~~~~~~~~~~ deallocate memory
  free(VGrid);
  free(PhiGrid);

  delete nGridDims;
}
