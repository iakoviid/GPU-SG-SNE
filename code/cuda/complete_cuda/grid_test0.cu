#include "relocateData.cuh"
#include "relocateData.hpp"
#include "gridding.cuh"
#include "gridding.hpp"
#include "utils_cuda.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "types.hpp"

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

template <class dataPoint>
void compair(dataPoint *const w, dataPoint *dv, int n, int d,
             const char *message, int same) {
  int bro = 1;
  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  for (int i = 0; i < 3; i++) {
    for (int j = 0; j < d; j++) {
      //printf("%lf - %lf   ", w[i * d + j], v[i + j * n]);
    }
  }
  printf("\n");

  printf(
      "----------------------------------Compair %s----------------------------"
      "-----------\n",
      message);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if (same == 0) {
        if (abs(w[i * d + j] - v[i + j * n]) < 0.01) {
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else if (i == 50) {
          bro = 0;
          cout << "Error "
               << "Host=" << w[i * d + j] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      } else {
        printf("%lf %lf  %lf\n",w[i + j * n],v[i + j * n],abs(w[i + j * n] - v[i + j * n]) );
        if (abs(w[i + j * n] - v[i + j * n]) < 0.01) {

          if(i<10){
          //cout <<"Succes" << "Host=" << w[i  + n*j] << " vs Cuda=" << v[i + j * n]<<endl;
        }
      } else if (i == 10) {
          bro = 0;
          cout << "Error "
               << "Host=" << w[i + j * n] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      }
    }
  }
  if (bro == 1) {
    if(same==1){
    for(int i=0;i<10;i++){
    //cout <<"Success" << "Host=" << w[i ] << " vs Cuda=" << v[i ]<<endl;
  }}
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Success~~~~~~~~~~~~~~~~~~~~~~~~\n");
  } else {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Error~~~~~~~~~~~~~~~~~~~~~~~~\n");
  }
  free(v);
}

__global__ void Normalize(coord* y,uint32_t nPts, uint32_t ng,uint32_t d,coord maxy){
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
for(int dim=0;dim<d;dim++){
    y[TID+dim*nPts] /= maxy;
    if (y[TID+dim*nPts] == 1) {
      y[TID+dim*nPts] = y[TID+dim*nPts] - 0.00000000000001;
    }
    y[TID+dim*nPts] *= (ng - 3);
  }
}
}
int main(int argc, char **argv) {
  srand(time(NULL));

  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  coord *y, *y_d;
  struct timeval t1, t2;
  double elapsedTime;

  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  uint32_t *ib, *cb, *ib_h, *cb_h;

  CUDA_CALL(cudaMallocManaged(&ib, ng *ng *sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, ng *ng* sizeof(uint32_t)));
  ib_h = (uint32_t *)calloc(ng, sizeof(uint32_t));
  cb_h = (uint32_t *)calloc(ng, sizeof(uint32_t));
  cudaMemset(cb, 0, ng*ng * sizeof(uint32_t));
  cudaMemset(ib, 0, ng*ng * sizeof(uint32_t));

  thrust::device_vector<uint32_t> iPerm(n);
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  double sum1 = 0;
  double sum2 = 0;
  uint32_t *icopy, *ccopy;
  icopy = (uint32_t *)malloc(ng*ng * sizeof(uint32_t));
  ccopy = (uint32_t *)malloc(ng *ng* sizeof(uint32_t));
  compair(y, y_d, n, d, "Y", 0);

  thrust::sequence(iPerm.begin(), iPerm.end());
  for (int j = 0; j < n; j++) {
    iPerm_h[j] = j;
  }

  gettimeofday(&t1, NULL);

  relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
   printf("CUDA time milliseconds %f\n", elapsedTime);
  sum1 += elapsedTime;
  gettimeofday(&t1, NULL);
  printf("Broo\n");
  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  printf("GOOD\n" );

  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
   printf("Host time milliseconds %f\n", elapsedTime);


  compair(y, y_d, n, d, "Y", 0);

  cudaMemcpy(icopy, ib, ng*ng * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ccopy, cb, ng*ng * sizeof(uint32_t), cudaMemcpyDeviceToHost);
for(int i=0;i<ng;i++){
  printf("ib_h[%d]=%d\n",i,ib_h[i] );

  for(int j=0;j<ng;j++){
    printf("icopy[%d]=%d  ",i*ng+j,icopy[i*ng+j] );


}
printf("\n" );

}


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
int szV = pow(ng + 2, d) * (d + 1);

coord *VGrid = (coord *)calloc(szV, sizeof(coord));
coord *VGrid_d;
CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));
initKernel<<<64, 256>>>(VGrid_d,(coord) 0, szV);
gettimeofday(&t1, NULL);

s2g2drbCpu(VGrid, y, VScat,ib_h,cb_h, ng + 2, 1, n, d, d + 1);
gettimeofday(&t2, NULL);

elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
printf("CPU Convolution time %lf\n",elapsedTime );
int Blocks=64;
int threads=32;
printf("blocks=%d  threads=%d\n",Blocks,threads );
Normalize<<<64,256>>>(y_d,n,ng+2,d,maxy);
compair(y, y_d, n, d, "Y", 0);

cudaDeviceSynchronize();
//compair(y, y_d, n, d, "Y", 0);

gettimeofday(&t1, NULL);


s2g2drb<<<64, 32>>>(VGrid_d, y_d, VScat_d,ib,cb, ng + 2, n, d, d + 1);

cudaDeviceSynchronize();
gettimeofday(&t2, NULL);

elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
printf("CUDA Convolution time %lf\n",elapsedTime );
compair(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "VGrid", 1);


/*
  sum2 += elapsedTime;
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
  int szV = pow(ng + 2, d) * (d + 1);

  coord *VGrid = (coord *)calloc(szV, sizeof(coord));
  coord *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));
  initKernel<<<64, 256>>>(VGrid_d,(coord) 0, szV);
  gettimeofday(&t1, NULL);
  s2g2drbCpu(VGrid, y, VScat,ib_h,cb_h, ng + 2, 1, n, d, d + 1);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("CPU Convolution time %lf\n",elapsedTime );
  int Blocks=64;
  int threads=32;
  printf("blocks=%d  threads=%d\n",Blocks,threads );
  Normalize<<<64,256>>>(y_d,n,ng+2,d,maxy);
  cudaDeviceSynchronize();
  compair(y, y_d, n, d, "Y", 0);

  gettimeofday(&t1, NULL);


  s2g2drb<<<Blocks, threads>>>(VGrid_d, y_d, VScat_d,ib,cb, ng + 2, n, d, d + 1);
  //s2g1d<<<32, 256>>>(VGrid_d, y_d, VScat_d, ng + 2, n, d, d + 1, maxy);

  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("CUDA Convolution time %lf\n",elapsedTime );

  compair(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "VGrid", 1);

  coord *VGrid_h = (coord *)calloc(szV, sizeof(coord));
  cudaMemcpy(VGrid_h, VGrid_d,szV* sizeof(coord), cudaMemcpyDeviceToHost);

/*
  for (size_t i = 0; i < 10; i++) {
    for(int j=0;j<d+1;j++){

    if((abs(VGrid_h[i  + n*j] - VGrid[i + j * n]) < 0.01) ){
      printf("Success VGrid=%lf  VGridh=%lf\n",VGrid[i  + n*j],VGrid_h[i  + n*j] );
    }
    else{
      printf("Error VGrid=%lf  VGridh=%lf\n",VGrid[i  + n*j],VGrid_h[i  + n*j] );


    }
  }
}
*/


  return 0;
}
