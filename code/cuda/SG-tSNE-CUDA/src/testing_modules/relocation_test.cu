#include "../relocateData.cuh"
#include "../relocateData.hpp"
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

template <class dataPoint>
dataPoint maxerror(dataPoint *const w, dataPoint *dv, int n, int d) {

  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  dataPoint maxError = 0;
  dataPoint avgError = 0;
  int pos = 0;

  for (int i = 0; i < n ; i++) {
    for(int j=0;j<d;j++){
    if ((v[i+j*n] - w[i*d+j]) * (v[i+j*n] - w[i*d+j]) > maxError) {
      maxError = (v[i+j*n] - w[i*d+j]) * (v[i+j*n] - w[i*d+j]);
      pos = i;
    }
    avgError += (v[i+j*n] - w[i*d+j]) * (v[i+j*n] - w[i*d+j]);
  }}

  printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d size=%d\n",
         maxError, pos, v[pos], w[pos], avgError / (n * d), n, n * d);
  free(v);
  return maxError;
}

int main(int argc, char **argv) {
  srand(time(NULL));

  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  int iterations=atoi(argv[4]);
  coord *y, *y_d;
  struct timeval t1, t2;
  double elapsedTime;

  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  uint32_t points= pow(ng, d);
  uint32_t *ib, *cb, *ib_h, *cb_h;
  CUDA_CALL(cudaMallocManaged(&ib, points * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, points * sizeof(uint32_t)));
  ib_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cb_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cudaMemset(cb, 0, points * sizeof(uint32_t));
  cudaMemset(ib, 0, points * sizeof(uint32_t));

  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  double sum1 = 0;
  double sum2 = 0;
  uint32_t *icopy, *ccopy;
  icopy = (uint32_t *)malloc(points * sizeof(uint32_t));
  ccopy = (uint32_t *)malloc(points * sizeof(uint32_t));
  uint32_t* iPerm;
  CUDA_CALL(cudaMallocManaged(&iPerm, n * sizeof(uint32_t)));

  double* timecpu=(double *)malloc(sizeof(double)*iterations);
  double* timegpu=(double *)malloc(sizeof(double)*iterations);
  for (int i = 0; i < iterations; i++) {
    // printf("i=%d\n",i );
    //compair(y, y_d, n, d, "Y", 0);
    y = generateRandomCoord(n, d);
    copydata(y, y_d, n, d);


    for (int j = 0; j < n; j++) {
      iPerm_h[j] = j;
    }
    cudaMemcpy(iPerm, iPerm_h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);

    gettimeofday(&t1, NULL);

    relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
    cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timegpu[i]=elapsedTime;
    sum1 += elapsedTime;
    gettimeofday(&t1, NULL);

    relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timecpu[i]=elapsedTime;
      //compair(y, y_d, n, d, "Y", 0);


    sum2 += elapsedTime;
    if(i!=iterations-1){
    free(y);}
  }
    //printf("===============> CUDA %lf  vs  HOST %lf: n=%d d=%d ng=%d iterations=%d\n",sum1/iterations,sum2/iterations,atoi(argv[1]),d,ng,iterations );
    //maxerror<coord>( y, y_d, n,d);
    for(int i=0;i<iterations;i++){
      printf("%lf ",timegpu[i] );
    }
    for(int i=0;i<iterations;i++){
      printf("%lf ",timecpu[i] );
    }
    printf("%lf %lf\n",sum1/iterations,sum2/iterations );

  free(icopy);
  free(ccopy);
  free(ib_h);
  free(cb_h);
  free(y);
  free(iPerm_h);
  cudaFree(y_d);
  cudaFree(iPerm);
  cudaFree(ib);
  cudaFree(cb);
  return 0;
}
