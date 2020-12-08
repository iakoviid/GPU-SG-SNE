#include "../gridding.cuh"
#include "../gridding.hpp"
#include "../relocateData.cuh"
#include "../relocateData.hpp"
#include "../non_periodic_conv.cuh"
#include "../non_periodic_conv.hpp"
#include "../nuconv.hpp"
#include "../nuconv.cuh"
#include "../Frep.hpp"
#include "../Frep.cuh"

#include <iostream>
//#include "tsne.cuh"

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
      dataPoint error=fabs(v[i+j*n] - w[i*d+j]);

      if (error > maxError) {
        maxError =error;
      pos = i;
    }
    avgError += error;
  }}

  printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d size=%d\n",
         maxError, pos, v[pos], w[pos*d], avgError / (n * d), n, n * d);
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
int main(int argc, char **argv) {
  srand(time(NULL));

  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  int iterations = atoi(argv[4]);
  coord *y, *y_d;

  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  coord *F_d,*F;
  CUDA_CALL(cudaMallocManaged(&F_d, (d)*n * sizeof(coord)));
  F=(coord * )malloc(sizeof(coord)*n*d);

  uint32_t *cb, *ib_h, *cb_h;
  int* ib;
  uint32_t points= pow(ng-1, d)+1;

  CUDA_CALL(cudaMallocManaged(&ib, points * sizeof(int)));
  CUDA_CALL(cudaMallocManaged(&cb, points * sizeof(uint32_t)));
  ib_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cb_h = (uint32_t *)calloc(points, sizeof(uint32_t));
  cudaMemset(cb, 0, points * sizeof(uint32_t));
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t* iPerm;
  CUDA_CALL(cudaMallocManaged(&iPerm, n * sizeof(uint32_t)));
  for (int i = 0; i < n; i++) {
    iPerm_h[i] = i;
  }
  cudaMemcpy(iPerm, iPerm_h, n * sizeof(uint32_t), cudaMemcpyHostToDevice);
  relocateCoarseGrid(y_d, iPerm, ib,  n, ng, d);
  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  maxerror(y, y_d, n, d);
  coord *VScat = generateRandomCoord(n, d + 1);
  coord *VScat_d;
  CUDA_CALL(cudaMallocManaged(&VScat_d, (d + 1) * n * sizeof(coord)));
  copydata(VScat, VScat_d, n, d + 1);
  int nVec=d+1;
  coord* Phi=(coord *)malloc(n*nVec*sizeof(coord));
  coord* Phi_d;
  CUDA_CALL(cudaMallocManaged(&Phi_d,nVec*n*sizeof(coord)));
  double timeInfo[3];
  for(int i=0;i<iterations;i++){

  nuconvCPU(Phi,y, VScat,ib_h, cb_h,n,d,nVec,1,ng,timeInfo);
  printf("CPU timeInfo:%lf %lf %lf \n",timeInfo[0],timeInfo[1],timeInfo[2] );
  nuconv(Phi_d,y_d, VScat_d, ib, n, d, nVec, ng,timeInfo);
  printf("timeInfo:%lf %lf %lf \n",timeInfo[0],timeInfo[1],timeInfo[2] );
  printf("Phi " );
  maxerror(Phi, Phi_d, n, d+1);
  printf("y  " );
  maxerror(y, y_d, n, d);

  zetaAndForce(F_d,y_d,n,d,Phi_d,iPerm);

  zetaAndForceCPU(F,y,Phi,iPerm_h,n,d);
  printf("F  " );
  maxerror(F, F_d, n, d);



  }

  cudaFree(cb);
  cudaFree(ib);
  cudaFree(Phi_d);
  cudaFree(VScat_d);
  cudaFree(y_d);
  cudaFree(iPerm);
  free(y);
  free(Phi);
  free(VScat);
  free(ib_h);
  free(cb_h);
  free(iPerm_h);
return 0;
}
