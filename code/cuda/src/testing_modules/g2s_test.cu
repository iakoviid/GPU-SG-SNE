
#include "../gridding.cuh"
#include "../gridding.hpp"
#include <limits>

#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../utils_cuda.cuh"

#include "../types.hpp"
coord *generateRandomCoord(int n, int d) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));

  for (int i = 0; i < n * d; i++)
    y[i] = ((coord)rand() / (RAND_MAX)) * 100;

  return y;
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
         maxError, pos, v[pos], w[pos*d], avgError / (n * d), n, n * d);
  free(v);
  return maxError;
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

void g2sCPU(coord *Phi, coord *V, coord *y, uint32_t ng,
          uint32_t n, uint32_t d, uint32_t m) {
  switch (d) {

  case 1:
    g2s1dCpu(Phi, V, y, ng + 2, n, d, m);
    break;
  case 2:
    g2s2dCpu(Phi, V, y, ng + 2, n, d, m);
    break;
  case 3:
    g2s3dCpu( Phi, V, y, ng+2, n, d, m );
    break;
  }
}
void g2s(coord *Phi, coord *V, coord *y, uint32_t ng,
          uint32_t n, uint32_t d, uint32_t m) {
  switch (d) {

  case 1:
    g2s1d<<<32, 256>>>(Phi, V, y, ng + 2, n, d, m);
    break;
  case 2:
    g2s2d<<<32, 256>>>(Phi, V, y, ng + 2, n, d, m);
    break;
  case 3:
    g2s3d<<<32, 256>>>( Phi, V, y, ng+2, n, d, m );
    break;
  }
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
int main(int argc, char **argv) {
  srand(time(NULL));

  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  int iterations = atoi(argv[4]);
  int m=d+1;
  int szV = pow(ng + 2, d) * m;
  coord *PhiGrid=generateRandomCoord(pow(ng + 2, d), d+1 );
  coord *PhiGrid_d;
  CUDA_CALL(cudaMallocManaged(&PhiGrid_d, szV * sizeof(coord)));
  cudaMemcpy(PhiGrid_d, PhiGrid, szV * sizeof(coord), cudaMemcpyHostToDevice);


  coord *y, *y_d;
  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  double *timecpu = (double *)malloc(iterations * sizeof(double));
  double *timegpu = (double *)malloc(iterations * sizeof(double));
  struct timeval t1, t2;
  double elapsedTime;
  double sum1 = 0;
  coord* PhiScat=(coord *)malloc(sizeof(coord)*n*(d+1));
  coord* PhiScat_d;
  CUDA_CALL(cudaMallocManaged(&PhiScat_d, (d+1) * n * sizeof(coord)));
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
maxerror<coord>( y, y_d, n,d);
  for(int i=0;i<iterations;i++){
    for (int j = 0; j < n*(d+1); j++) {PhiScat[j] = 0;}
    gettimeofday(&t1, NULL);
	   g2sCPU(PhiScat, PhiGrid, y, ng,n, d, d+1);

    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timecpu[i] = elapsedTime;
    sum1 += elapsedTime;
  }

  double sum2 = 0;
  for(int i=0;i<iterations;i++){
    initKernel<<<64, 256>>>(PhiScat_d, (coord)0,n*(d+1) );
	cudaDeviceSynchronize();

   	 gettimeofday(&t1, NULL);
	    g2s(PhiScat_d, PhiGrid_d, y_d, ng,n, d, d+1);
	 cudaDeviceSynchronize();
    gettimeofday(&t2, NULL);

    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    timegpu[i] = elapsedTime;

    sum2 += elapsedTime;

  }

  maxerror<coord>( PhiScat, PhiScat_d, n,d+1);
  //coord *PhiScat_c=(coord* )malloc(sizeof(coord)*n*(d+1));
  //cudaMemcpy(PhiScat_c, PhiScat_d, (d+1) * n * sizeof(coord), cudaMemcpyDeviceToHost);

 for (int i = 0; i < iterations; i++) {
    printf("%lf ", timegpu[i]);
  }
 for (int i = 0; i < iterations; i++) {
    printf("%lf ", timecpu[i]);
  }
 printf("%lf %lf",sum2/iterations,sum1/iterations);
 printf("\n" );
 free(y);
 cudaFree(y_d);
 cudaFree(PhiScat_d);
 free(PhiScat);
 free(timecpu);
 free(timegpu);
 cudaFree(PhiGrid_d);
 cudaFree(PhiGrid);

return 0;
}
