#include "../gridding.cuh"
#include "../gridding.hpp"
#include "../relocateData.cuh"
#include "../relocateData.hpp"
#include "../utils_cuda.cuh"
#include "../Frep.cuh"
#include "../Frep.hpp"

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../types.hpp"

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
         maxError, pos, v[pos], w[pos*d], avgError / (n * d), n, n * d);
  free(v);
  return maxError;
}

coord *generateRandomCoord(int n, int d) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));
  for(int i=0;i<n;i++){
    for(int j=0;j<d;j++){
      scanf("%lf %lf\n", &y[i*d], &y[i*d+1]);

    }
  }
  //for (int i = 0; i < n * d; i++)
    //y[i] = ((coord)rand() / (RAND_MAX)) * 100;

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

int main(int argc, char **argv) {
  srand(time(NULL));

  int n = atoi(argv[1]);
  int d = atoi(argv[2]);
  //int ng = atoi(argv[3]);
  int iterations=atoi(argv[4]);
  coord *y, *y_d;
  struct timeval t1, t2;
  double elapsedTime;
  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  coord *Frep, *Frep_d;
  CUDA_CALL(cudaMallocManaged(&Frep_d, (d)*n * sizeof(coord)));
  Frep=(coord *)malloc(n*d*sizeof(coord));
  for(int i=0;i<iterations;i++){
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
/*
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;

  coord h = maxy / (ng - 1 - std::numeric_limits<double>::epsilon());
*/
  coord h=0.7;
  coord zeta1;
  coord zeta2;
  double timeInfo[7];
  double timeInfo1[6];
  //printf("times Frep Reloc S2G G2G G2S\n" );
  gettimeofday(&t1, NULL);

  zeta1=computeFrepulsive_interpCPU(Frep, y,  n,  d, h,1,timeInfo1);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  printf("Host time milliseconds %lf %lf %lf %lf %lf %lf %lf\n", elapsedTime,timeInfo1[0],timeInfo1[1],timeInfo1[2],timeInfo1[3],timeInfo1[4],timeInfo1[5]);
  gettimeofday(&t1, NULL);

  zeta2= computeFrepulsive_interp(Frep_d, y_d, n, d, h,timeInfo);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  double time1=elapsedTime;
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;      // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  printf("CUDA %lf %lf %lf %lf %lf %lf %lf %lf\n", elapsedTime,timeInfo[0],timeInfo[1],timeInfo[2],timeInfo[3],timeInfo[4],timeInfo[5],timeInfo[6]);

  //printf("CUDA time milliseconds %lf %lf %lf %lf %lf %lf\n", elapsedTime,timeInfo[0],timeInfo[1],timeInfo[2],timeInfo[3],timeInfo[4]);
  printf("speedup  %lf %lf %lf %lf %lf %lf %lf\n",time1/elapsedTime,timeInfo1[0]/timeInfo[0],timeInfo1[1]/timeInfo[1],timeInfo1[2]/timeInfo[2],timeInfo1[3]/timeInfo[3],timeInfo1[4]/timeInfo[4],timeInfo1[5]/timeInfo[5] );
  printf("zeta1=%lf vs zeta2=%lf\n",zeta1,zeta2 );
  maxerror(Frep, Frep_d, n, d);
  free(y);
  }
  cudaFree(y_d);
  free(Frep);
  cudaFree(Frep_d);
return 0;
}

/*
107--------------------------------------
zeta=3460151.066943 sum=0.000000
zeta=3460151.066943 sum=0.000000

108--------------------------------------
here
zeta=102666372394.358673 sum=0.000000
108--------------------------------------
zeta=3394236.432173 sum=0.000000
*/
