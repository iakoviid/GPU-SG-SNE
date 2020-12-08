
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
template <class dataPoint>
dataPoint maxerror(dataPoint *const w, dataPoint *dv, int n, int d,
              const char *message) {

  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  dataPoint maxError = 0;
  dataPoint avgError = 0;
  int s=20;
  for(int j=0;j<d;j++){
  for (int i = 0; i < n ; i++) {

    if((v[i*d+j] - w[i*d+j]) * (v[i*d+j] - w[i*d+j]>0.1 && s>0))
      {printf("i=%d d=%d ",i,j );
        printf("%lf vs %lf\n",v[i*d+j],w[i*d+j] );
        s--;
    }

    if ((v[i*d+j] - w[i*d+j]) * (v[i*d+j] - w[i*d+j]) > maxError) {
      maxError = (v[i*d+j] - w[i*d+j]) * (v[i*d+j] - w[i*d+j]);

    }

    avgError += (v[i*d+j] - w[i*d+j]) * (v[i*d+j] - w[i*d+j]);
  }}

  //printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d size=%d\n",maxError, pos, v[pos], w[pos], avgError / (n * d), n, n * d);
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

void testgridding(uint32_t *ib, uint32_t *cb, int *ib_d, uint32_t *cb_d,
                    coord *VGrid, coord *y, coord *VScat, coord *VGrid_d,
                    coord *y_d, coord *VScat_d, uint32_t n, uint32_t d,
                    uint32_t ng, int iterations) {

  double *timecpu = (double *)malloc(iterations * sizeof(double));
  double *timegpu = (double *)malloc(iterations * sizeof(double));
  double *timegpuwarp = (double *)malloc(iterations * sizeof(double));

  int szV = pow(ng + 2, d) * (d + 1);
double errorsimple=0;

  struct timeval t1, t2;
  double elapsedTime;

  for (int i = 0; i < iterations; i++) {
    for (int j = 0; j < szV; j++) {VGrid[j] = 0;}

    gettimeofday(&t1, NULL);
    s2gCPU(VGrid, y, VScat, ng, n, d);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timecpu[i] = elapsedTime;

    initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
    cudaDeviceSynchronize();
    gettimeofday(&t1, NULL);
    s2g(VGrid_d, y_d, VScat_d, ng + 2, n, d, d + 1);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timegpu[i] = elapsedTime;
    errorsimple=maxerror(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "Gridding simple");


    initKernel<<<64, 256>>>(VGrid_d, (coord)0, szV);
    cudaDeviceSynchronize();

    gettimeofday(&t1, NULL);

    s2gwarp(VGrid_d, y_d, VScat_d, ib_d, ng + 2, n, d, d + 1);

    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timegpuwarp[i] = elapsedTime;

  }
  double errorwarp=maxerror(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "Gridding warp");

  printf("n=%d d=%d ng=%d\n",n,d,ng );
  printf("cpu: ");
  for (int i = 0; i < iterations; i++) {
    printf("%lf ", timecpu[i]);
  }
  printf("\n");
  printf("gpuwarp: ");
  for (int i = 0; i < iterations; i++) {
    printf("%lf ", timegpuwarp[i]);
  }
  printf("\n");
  printf("gpu: ");
  for (int i = 0; i < iterations; i++) {
    printf("%lf ", timegpu[i]);
  }

  printf("\n");
  printf("Errorsimple=%lf\n",errorsimple );
  //printf("Errorsim=%lf\n",errorsim );

  printf("Errorwarp=%lf\n",errorwarp );
  printf("\n" );
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
  uint32_t  *cb, *ib_h, *cb_h;
  int * ib;
  uint32_t points= pow(ng-1, d)+1;
  CUDA_CALL(cudaMallocManaged(&ib, points * sizeof(int)));
  CUDA_CALL(cudaMallocManaged(&cb, points * sizeof(uint32_t)));
  ib_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cb_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cudaMemset(cb, 0, points * sizeof(uint32_t));

  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));


  for (int j = 0; j < n; j++) {
    iPerm_h[j] = j;
  }
  uint32_t *iPerm;
  CUDA_CALL(cudaMallocManaged(&iPerm, n * sizeof(uint32_t)));
  cudaMemcpy(iPerm, iPerm_h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

  relocateCoarseGrid(y_d, iPerm, ib, n, ng, d);

  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
/*
      uint32_t* ib_c = (uint32_t *)calloc(points, sizeof(uint32_t));
      uint32_t* cb_c = (uint32_t *)calloc(points, sizeof(uint32_t));
      cudaMemcpy(ib_c, ib, points * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      cudaMemcpy(cb_c, cb, points * sizeof(uint32_t), cudaMemcpyDeviceToHost);
      for(int i=0;i<points;i++){
        if(ib_c[i]!=ib_h[i]){printf("Error " );}
          printf("i=%d gpu= %d cpu= %d\n",i,ib_c[i],ib_h[i] );
      }
      */
  //cudaMemcpy(ib, ib_h, points * sizeof(uint32_t), cudaMemcpyHostToDevice);
  //cudaMemcpy(cb, cb_h, points * sizeof(uint32_t), cudaMemcpyHostToDevice);
  //cudaMemcpy(ib_h, ib, points * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  //cudaMemcpy(cb_h, cb, points * sizeof(uint32_t), cudaMemcpyDeviceToHost);

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
  testgridding(ib_h, cb_h, ib, cb, VGrid, y, VScat, VGrid_d, y_d, VScat_d, n,
                 d, ng, iterations);
  free(y);
  free(ib_h);
  free(cb_h);
  free(iPerm_h);
  free(VGrid);
  free(VScat);
  cudaFree(y_d);
  cudaFree(ib);
  cudaFree(cb);
  cudaFree(iPerm);
  cudaFree(VGrid_d);
  cudaFree(VScat_d);

  cudaDeviceReset();
  return 0;
}
