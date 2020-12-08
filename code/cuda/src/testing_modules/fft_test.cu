#include "../non_periodic_conv.hpp"
#include "../non_periodic_convD.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
#include <limits>
using namespace std;
#include "../types.hpp"
#include "../utils_cuda.cuh"
void connopad(coord *const PhiGrid, const coord *const VGrid, const coord h,
              uint32_t *const nGridDims, const uint32_t nVec,
              const uint32_t nDim, const uint32_t nProc) {
  switch (nDim) {

  case 1:
    conv1dnopad(PhiGrid, VGrid, h, nGridDims, nVec, nDim, nProc);
    break;
  case 2:
    conv2dnopad(PhiGrid, VGrid, h, nGridDims, nVec, nDim, nProc);
    break;
  case 3:
    conv3dnopad(PhiGrid, VGrid, h, nGridDims, nVec, nDim, nProc);
    break;
  }
}

void convnopadCuda(coord* PhiGrid_d,coord* VGrid_d,coord h,uint32_t* nGridDims,int nVec,int d){
  switch (d) {

  case 1:
    conv1dnopadcuda(PhiGrid_d, VGrid_d, h, nGridDims, d + 1, d);
    break;
  case 2:
    conv2dnopadcuda(PhiGrid_d, VGrid_d, h, nGridDims, d + 1, d);
    break;
  case 3:
    conv3dnopadcuda(PhiGrid_d, VGrid_d, h, nGridDims, d + 1, d);
    break;
  }

}
template <class dataPoint>
void maxerror(dataPoint *const w, dataPoint *dv, int n, int d,
             const char *message){
               printf(
                   "----------------------------------Compair %s----------------------------"
                   "-----------\n",
                   message);
               dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
               cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
               dataPoint maxError=0;
               dataPoint avgError=0;
               int pos=0;
               for(int i=0;i<n*d;i++){

                 dataPoint error=fabs(v[i+j*n] - w[i*d+j]);
                 if (error > maxError) {
                   maxError =error;
                   pos=i;

                 }
                 avgError += error;

               }


        printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d size=%d\n",maxError,pos,v[pos],w[pos],avgError/(n*d),n,n*d );
free(v);

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
        //printf("%lf %lf  %lf\n",w[i + j * n],v[i + j * n],abs(w[i + j * n] - v[i + j * n]) );
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

void testfft(coord* PhiGrid,coord* PhiGrid_d,coord* VGrid,coord* VGrid_d,coord h,uint32_t d, uint32_t* nGridDims,uint32_t ng,uint32_t iterations){
  uint32_t tpoints=pow(ng + 2, d);
  int szV = pow(ng + 2, d) * (d + 1);

  struct timeval t1, t2;
  double elapsedTime;
  double sum1=0;
  double* timecpu=(double*) malloc(iterations*sizeof(double));
  double* timegpu=(double*) malloc(iterations*sizeof(double));
  for(int i=0;i<iterations;i++){
  gettimeofday(&t1, NULL);

  connopad(PhiGrid, VGrid, h, nGridDims, d+1, d, 1);
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  timecpu[i]=elapsedTime;
  sum1+=elapsedTime;
  }

  //printf("cpu Avg time %lf vs ",sum1/iterations );

  double sum2=0;

  for(int i=0;i<iterations;i++){
  initKernel<coord><<<32,256>>>(PhiGrid_d,0,szV);
  cudaDeviceSynchronize();

  gettimeofday(&t1, NULL);

  convnopadCuda(PhiGrid_d, VGrid_d, h, nGridDims, d+1, d);
  //convnopadCuda(PhiGrid_d, VGrid_d, h, nGridDims, d+1, d);

  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  sum2+=elapsedTime;
  timegpu[i]=elapsedTime;

}
  //printf("CUDA Avg time %lf: ng%d d=%d \n",sum2/iterations,ng,d );
  //compair(PhiGrid, PhiGrid_d, tpoints, d + 1, "PhiFFT", 1);
 maxerror(PhiGrid, PhiGrid_d, tpoints, d + 1, "PhiFFT");
  for(int i=0;i<iterations;i++){
    printf("%lf ",timecpu[i] );
  }


  for(int i=0;i<iterations;i++){
    printf("%lf ",timegpu[i] );
  }
    printf("%lf %lf",sum1/iterations,sum2/iterations );
  printf("\n" );
  free(timecpu);
  free(timegpu);

}
void initAndTest(int ng,int iterations, int d){
  uint32_t tpoints = pow(ng + 2, d);
  coord h = 1 / (ng - 1 - std::numeric_limits<double>::epsilon());
  uint32_t * nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = ng + 2;
  }
  int szV = pow(ng + 2, d) * (d + 1);
  coord *VGrid = generateRandomCoord( pow(ng + 2, d), (d + 1));
  coord *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));
  cudaMemcpy(VGrid_d, VGrid,szV* sizeof(coord), cudaMemcpyHostToDevice);
  coord *PhiGrid = static_cast<coord *>(calloc(szV, sizeof(coord)));
  coord *PhiGrid_d;
  CUDA_CALL(cudaMallocManaged(&PhiGrid_d, szV * sizeof(coord)));


  testfft( PhiGrid, PhiGrid_d, VGrid, VGrid_d, h, d,  nGridDims,ng,iterations);
  free(VGrid);
  cudaFree(VGrid_d);
  free(PhiGrid);
  cudaFree(PhiGrid_d);

}
int main(int argc, char **argv) {
    srand(time(NULL));
    int iterations=atoi(argv[3]);
    int d = atoi(argv[2]);
    int ng = atoi(argv[1]);
    //int sizes=atoi(argv[4]);
    //for(int i=ng;i<sizes;i+=20){
      //printf("========================Test %d========================\n",i );
      initAndTest(ng,iterations,d);
    //}


    return 0;

  }
