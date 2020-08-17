#include "relocateData.cuh"
#include "relocateData.hpp"
#include <iostream>
#include <sys/time.h>
#include <stdio.h>
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
void compair(dataPoint *const w, dataPoint *dv, int n, int d,const  char *message,
             int same) {
  int bro = 1;
  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  for(int i=0;i<3;i++){
    for(int j=0;j<d;j++){
      //printf("%lf - %lf   ",w[i*d+j],v[i+j*n] );
    }
  }
printf("\n" );

  printf(
      "----------------------------------Compair %s----------------------------"
      "-----------\n",
      message);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if (same == 0) {
        if (abs(w[i * d + j] - v[i + j * n]) < 0.01) {
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else if(i==50 ){
          bro = 0;
          cout << "Error "
               << "Host=" << w[i * d + j] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      } else {
        if (abs(w[i + j * n] - v[i + j * n]) < 0.01 ){
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else if(i==50 ) {
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
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Success~~~~~~~~~~~~~~~~~~~~~~~~\n");
  } else {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Error~~~~~~~~~~~~~~~~~~~~~~~~\n");
  }
  free(v);
}
int main(int argc, char **argv) {
  srand(time(NULL));

  int n = 1 << atoi(argv[1]);
  int d = 2;
  int ng = 14;
  coord *y, *y_d;
  struct timeval t1, t2;
  double elapsedTime;

  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  uint32_t *ib, *cb, *ib_h, *cb_h;
  CUDA_CALL(cudaMallocManaged(&ib, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, ng * sizeof(uint32_t)));
  ib_h=(uint32_t *)malloc(ng*sizeof(uint32_t));
  cb_h=(uint32_t *)malloc(ng*sizeof(uint32_t));
  thrust::device_vector<uint32_t> iPerm(n);
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  double sum1=0;
  double sum2=0;
  uint32_t *icopy,*ccopy;
  icopy=(uint32_t *)malloc(ng*sizeof(uint32_t));
  ccopy=(uint32_t *)malloc(ng*sizeof(uint32_t));
  for(int i=0;i<10;i++){
    //printf("i=%d\n",i );
    y = generateRandomCoord(n, d);
    copydata(y, y_d, n, d);
    compair(y, y_d, n, d, "Y", 0);

  thrust::sequence(iPerm.begin(), iPerm.end());

  for (int j = 0; j < n; j++) {
    iPerm_h[j] = j;
  }
  gettimeofday(&t1, NULL);

  relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  //printf("CUDA time milliseconds %f\n", elapsedTime);
  sum1+=elapsedTime;
  gettimeofday(&t1, NULL);

  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  //printf("Host time milliseconds %f\n", elapsedTime);
  compair(y, y_d, n, d, "Y", 0);
  cudaMemcpy(icopy, ib, ng * sizeof(uint32_t), cudaMemcpyDeviceToHost);
  cudaMemcpy(ccopy, cb, ng * sizeof(uint32_t), cudaMemcpyDeviceToHost);

  for(int x=0;x<ng;x++){
    printf(" ib_h=%d   ib=%d   cb_h=%d  cb=%d \n",ib_h[x],icopy[x],cb_h[x],ccopy[x] );
  }

  sum2+=elapsedTime;
  free(y);
  }
  printf("CUDA %lf  vs  HOST %lf\n",sum1/10,sum2/10 );



  return 0;
}
