
#include "../gridding.cuh"
#include "../gridding.hpp"
#include "../relocateData.cuh"
#include "../relocateData.hpp"
#include "../timer.h"
#include "../utils_cuda.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../types.hpp"
coord *generateRandomCoord(int n, int d) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));

  for (int i = 0; i < n * d; i++)
    y[i] = ((coord)rand() / (RAND_MAX)) * 100;

  return y;
}

template <class dataPoint>
void copydata(dataPoint *const w, dataPoint *dw, int n, int d) {
  dataPoint *v = (dataPoint *)malloc(sizeof(dataPoint) * n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {

      v[i + n * j] = w[i * d + j];
    }
  }
  cudaMemcpy(dw, v, d * n * sizeof(dataPoint), cudaMemcpyHostToDevice);
  free(v);
  return;
}

__global__ void Normalize(coord *y, uint32_t nPts, uint32_t ng, uint32_t d,
                          coord maxy) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (int dim = 0; dim < d; dim++) {
      y[TID + dim * nPts] /= maxy;
      if (y[TID + dim * nPts] == 1) {
        y[TID + dim * nPts] = y[TID + dim * nPts] - 0.00000000000001;
      }
      y[TID + dim * nPts] *= (ng - 3);
    }
  }
}
void Reloc(int *ib, coord* y, coord* y_d, uint32_t n, uint32_t d, uint32_t ng) {
  uint32_t points = pow(ng - 1, d) + 1;
  CUDA_CALL(cudaMallocManaged(&ib, points * sizeof(int)));
  uint32_t *ib_h, *cb_h;
  ib_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cb_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  for (int j = 0; j < n; j++) {
    iPerm_h[j] = j;
  }
  uint32_t *iPerm;
  CUDA_CALL(cudaMallocManaged(&iPerm, n * sizeof(uint32_t)));
  cudaMemcpy(iPerm, iPerm_h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

  relocateCoarseGrid(y_d, iPerm, ib, n, ng, d);
  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  free(ib_h);
  free(cb_h);
  free(iPerm_h);
  cudaFree(iPerm);
}
template <class dataPoint>
dataPoint maxerror(dataPoint *const w, dataPoint *dv, int n, int d,
                   const char *message) {

  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  dataPoint maxError = 0;
  dataPoint avgError = 0;
  int s = 20;
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < n; i++) {

      coord er = (v[i * d + j] - w[i * d + j]) * (v[i * d + j] - w[i * d + j]);
      if (er > 0.001 && s > 0) {
        printf("Error ");
        printf("i=%d d=%d ", i, j);
        printf("%lf vs %lf\n", v[i * d + j], w[i * d + j]);
        s--;
      }

      if (er > maxError) {
        maxError = er;
      }
      avgError += er;
    }
  }

  free(v);
  return maxError;
}
void s2gCPU(coord *VGrid, coord *y, coord *VScat, uint32_t ng, uint32_t n,
            uint32_t d) {
  switch (d) {

  case 1:
    s2g1dCpu(VGrid, y, VScat, ng , 1, n, d, d + 1);
    break;
  case 2:
    s2g2dCpu(VGrid, y, VScat, ng , 1, n, d, d + 1);
    break;
  case 3:
    s2g3dCpu(VGrid, y, VScat, ng , 1, n, d, d + 1);
    break;
  }
}

int main(int argc, char **argv) {
  struct GpuTimer timer;

  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  int iterations = atoi(argv[4]);
  coord *y, *y_d;
  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);

  int *ib;
  Reloc(ib, y, y_d, n, d, ng);
  coord *VScat = generateRandomCoord(n, d + 1);
  coord *VScat_d;
  CUDA_CALL(cudaMallocManaged(&VScat_d, (d + 1) * n * sizeof(coord)));
  copydata(VScat, VScat_d, n, d + 1);
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for (int i = 0; i < n * d; i++) {
    y[i] /= maxy;

    if (1 == y[i])
      y[i] = y[i] - std::numeric_limits<coord>::epsilon();

    y[i] *= (ng - 1);
  }
  Normalize<<<64, 256>>>(y_d, n, ng + 2, d, maxy);
  int szV = pow(ng + 2, d) * (d + 1);

  coord *VGrid = (coord *)calloc(szV, sizeof(coord));
  coord *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));

  for (int i = 0; i < iterations; i++) {
    for (int j = 0; j < szV; j++) {VGrid[j] = 0;}
    s2gCPU(VGrid, y, VScat, ng+2, n, d);
    initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
    cudaDeviceSynchronize();
    timer.Start();

    switch (d) {

    case 1:
      s2g1d<<<64, 512>>>(VGrid_d, y_d, VScat_d, ng+2, n, d, d+1);
      break;
    case 2:
      s2g2d<<<64, 512>>>(VGrid_d, y_d, VScat_d, ng+2, n, d, d+1);
      break;
    case 3:
      s2g3d<<<64, 512>>>(VGrid_d, y_d, VScat_d, ng+2, n, d, d+1);
      break;
    }
    timer.Stop();
    printf("time simple %f \n",timer.Elapsed() );
    printf("Errorsimple %lf\n",maxerror(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "Gridding simple"));
    initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
    cudaDeviceSynchronize();
    timer.Start();

    switch (d) {

    case 1:
      s2g1drbwarp<<<Gridsz, BlockSizeWarp1D>>>(VGrid_d, y_d, VScat_d,ib, ng+2, n, d, d+1);
      break;
    case 2:
      s2g2drbwarpshared<<<Gridsz, BlockSizeWarp2Dshared>>>(VGrid_d, y_d, VScat_d,ib, ng+2, n, d, d+1);
      break;
    case 3:
      s2g3drbwarp<<<Gridsz, BlockSizeWarp3D>>>(VGrid_d, y_d, VScat_d,ib, ng+2, n, d, d+1);
      break;
    }
    timer.Stop();
    printf("time ord %f \n",timer.Elapsed() );
    printf("Errorwarp %lf\n",maxerror(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "Gridding simple"));


  }
  free(y);
  cudaFree(y_d);
  cudaFree(ib);
  cudaFree(VScat_d);
  free(VScat);
  cudaFree(VGrid_d);
  free(VGrid);

  return 0;
}
