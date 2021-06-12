
#include "../gridding.cuh"
#include "../gridding.hpp"
#include "../relocateData.cuh"
#include "../relocateData.hpp"
#include "../utils_cuda.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../types.hpp"
#include "../timer.h"

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
void s2gCPU(coord *VGrid, coord *y, coord *VScat, uint32_t ng, uint32_t n,
            uint32_t d) {
  switch (d) {

  case 1:
    s2g1dCpu(VGrid, y, VScat, ng + 2, 1, n, d, d + 1);
    break;
  case 2:
    s2g2dCpu(VGrid, y, VScat, ng + 2, 1, n, d, d + 1);
    break;
  case 3:
    s2g3dCpu(VGrid, y, VScat, ng + 2, 1, n, d, d + 1);
    break;
  }
}

void testgridding(uint32_t* ib_old,int *ib, uint32_t *cb, int *ib_d,
                  coord *VGrid, coord *y, coord *VScat, coord *VGrid_d,
                  coord *y_d, coord *VScat_d, uint32_t n, uint32_t d,
                  uint32_t ng, int iterations) {
                    struct GpuTimer timer;

  float *timecpu = (float *)malloc(iterations * sizeof(float));
  float *timegpu = (float *)malloc(iterations * sizeof(float));
  float *timegpuwarp = (float *)malloc(iterations * sizeof(float));

  int szV = pow(ng + 2, d) * (d + 1);

  struct timeval t1, t2;
  double elapsedTime;

  for (int i = 0; i < 1; i++) {
    for (int j = 0; j < szV; j++) {
      VGrid[j] = 0;
    }

    gettimeofday(&t1, NULL);
    s2gCPU(VGrid, y, VScat, ng, n, d);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timecpu[i] = elapsedTime;
  }

  for (int i = 0; i < iterations; i++) {
    initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
    cudaDeviceSynchronize();

    timer.Start();

    s2g(VGrid_d, y_d, VScat_d, ng + 2, n, d, d + 1);
    //s2g1d<<<64, 512>>>(VGrid_d, y_d, VScat_d, ng+2, n, d,d+1);
    timer.Stop();
    timegpu[i] = timer.Elapsed();

  }
  double errorsimple =
      maxerror(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "Gridding simple");

  for (int i = 0; i < iterations; i++) {
    initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
    cudaDeviceSynchronize();
    timer.Start();


    s2gwarp(VGrid_d, y_d, VScat_d, ib_d, ng + 2, n, d, d + 1);
    //s2g1drbwarpOld<<<Gridsz, BlockSizeWarp1D>>>(VGrid_d, y_d, VScat_d, ib_old, ng+2, n, d, d+1);

    timer.Stop();
    double el= timer.Elapsed();
    printf("time %lf \n",el );
    timegpuwarp[i] =  timer.Elapsed();


  }

  double errorwarp =
      maxerror(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "Gridding warp");

  if (d == 2) {
    coord *VGrid2 = (coord *)calloc(szV, sizeof(coord));
    cpuwarpsim(VGrid2, y, VScat, ib, cb, ng + 2, n, d, d + 1);
    coord maxError = 0;
    for (int i = 0; i < szV; i++) {
      // if(i<5){printf("VGrid2[i]=%lf  VGrid[i]=%lf\n",VGrid2[i],VGrid[i] );}
      coord er = (VGrid2[i] - VGrid[i]) * (VGrid2[i] - VGrid[i]);
      if (er > maxError) {
        maxError = er;
      }
    }
    printf("sim er=%lf\n", maxError);
    free(VGrid2);
  }
  if (d == 3) {
    coord *VGrid2 = (coord *)calloc(szV, sizeof(coord));

    cpuwarpsim3d(VGrid2, y, VScat, ib, cb, ng + 2, n, d, d + 1);
    coord maxError = 0;
    for (int i = 0; i < szV; i++) {
      coord er = (VGrid2[i] - VGrid[i]) * (VGrid2[i] - VGrid[i]);
      if (er > maxError) {
        maxError = er;
      }
    }
    printf("sim er=%lf\n", maxError);
    free(VGrid2);
  }

  printf("Errorsimple=%lf\n", errorsimple);
  printf("Errorwarp=%lf\n", errorwarp);

  printf("cpu: ");
  for (int i = 0; i < 1; i++) {
    printf("%f ", timecpu[i]);
  }
  printf("\n");

  printf("gpuwarp: ");
  for (int i = 0; i < iterations; i++) {
    printf("%f ", timegpuwarp[i]);
  }
  printf("\n");
  printf("gpu: ");
  for (int i = 0; i < iterations; i++) {
    printf("%f ", timegpu[i]);
  }
  printf("\n");
  free(timecpu);
  free(timegpu);

  free(timegpuwarp);
}

int main(int argc, char **argv) {
  CUDA_CALL(cudaDeviceSetSharedMemConfig(cudaSharedMemBankSizeEightByte));
  CUDA_CALL(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));
  srand(time(NULL));

  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  int iterations = atoi(argv[4]);

  coord *y, *y_d;

  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  uint32_t  * ib_old,*ib_h, *cb_h;
  int *ib;
  uint32_t points = pow(ng - 1, d) + 1;
  CUDA_CALL(cudaMallocManaged(&ib, points * sizeof(int)));
  ib_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cb_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  CUDA_CALL(cudaMallocManaged(&ib_old, points * sizeof(uint32_t)));

  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));

  for (int j = 0; j < n; j++) {
    iPerm_h[j] = j;
  }
  uint32_t *iPerm;
  CUDA_CALL(cudaMallocManaged(&iPerm, n * sizeof(uint32_t)));
  cudaMemcpy(iPerm, iPerm_h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

  relocateCoarseGrid(y_d, iPerm, ib, n, ng, d);

  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  int *ib_test = (int *)calloc(points, sizeof(int));

  cudaMemcpy(ib_test, ib, points * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(ib_old, ib_h, points * sizeof(uint32_t), cudaMemcpyHostToDevice);

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

  initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
  printf("n=%d ng=%d d=%d\n",n,ng,d );
  testgridding(ib_old,ib_test, cb_h, ib, VGrid, y, VScat, VGrid_d, y_d, VScat_d, n,
               d, ng, iterations);
  free(y);
  free(ib_h);
  free(cb_h);
  free(iPerm_h);
  free(VGrid);
  free(VScat);
  free(ib_test);
  cudaFree(y_d);
  cudaFree(ib);
  cudaFree(iPerm);
  cudaFree(VGrid_d);
  cudaFree(VScat_d);
  cudaFree(ib_old);
  cudaDeviceReset();
  return 0;
}
