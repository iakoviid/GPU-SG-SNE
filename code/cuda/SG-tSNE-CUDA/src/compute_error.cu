#include <thrust/device_vector.h>
#include <thrust/reduce.h>
#include "common.cuh"
#include "utils_cuda.cuh"
#include <cmath>

/*Not optimazed for testing purposes*/
template <class dataPoint>
__global__ void Frep_exactKernel(volatile dataPoint *__restrict__ frep,
                            const dataPoint *const pointsX,
                            volatile dataPoint *__restrict__ zetaVec,
                            const int num_points, const int d) {

  register dataPoint Yi[3] = {0};
  register dataPoint Yj[3] = {0};
  register dataPoint dist = 0.0;
  for (register int i = threadIdx.x + blockIdx.x * blockDim.x; i < num_points;
       i += gridDim.x * blockDim.x) {
    for (int dd = 0; dd < d; dd++) {
      Yi[dd] = pointsX[i + dd * num_points];
    }
    for (int j = 0; j < num_points; j++) {

      if (i != j) {

        dist = 0.0;
        for (int dd = 0; dd < d; dd++) {
          Yj[dd] = pointsX[j + dd * num_points];
          dist += (Yj[dd] - Yi[dd]) * (Yj[dd] - Yi[dd]);
        }

        for (int dd = 0; dd < d; dd++) {
          frep[i + dd * num_points] +=
              (Yi[dd] - Yj[dd]) / ((1 + dist) * (1 + dist));
        }

        zetaVec[i] += 1.0 / (1.0 + dist);
      }
    }
  }
}
template <class dataPoint>
dataPoint computeFrepulsive_exact(dataPoint *frep, dataPoint *pointsX, int n,
                                  int d) {

  thrust::device_vector<dataPoint> zetaVec(n);
  int threads = 1024;
  int Blocks = 64;
  Frep_exactKernel<<<Blocks, threads>>>(
      frep, pointsX, thrust::raw_pointer_cast(zetaVec.data()), n, d);
  dataPoint z = thrust::reduce(zetaVec.begin(), zetaVec.end());
  ArrayScale<<<Blocks, threads>>>(frep, 1 / z, n * d);
  return z;
}
template <class dataPoint>
__global__ void distance(dataPoint* x,dataPoint *y,int n, int d){

  for (register int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
         dataPoint dist=0;
         for (int dd = 0; dd < d; dd++) {
           dist+=(x[i+dd*n]-y[i+dd*n])*(x[i+dd*n]-y[i+dd*n]);
         }
         x[i]=dist;
       }
}
template <class dataPoint>
__global__ void vectornorm(dataPoint* x,dataPoint *y,int n, int d){

  for (register int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
         dataPoint dist=0;
         for (int dd = 0; dd < d; dd++) {
           dist+=(x[i+dd*n])*(x[i+dd*n]);
         }
         y[i]=dist;
       }
}
#include "utils.cuh"
template <class dataPoint>
dataPoint computeError(dataPoint *frep, dataPoint *y, int n, int d) {
  int threads = 1024;
  int Blocks = 64;
  thrust::device_vector<dataPoint> frep_exact(n * d);
  thrust::device_vector<dataPoint> normas(n);
  computeFrepulsive_exact(thrust::raw_pointer_cast(frep_exact.data()), y, n, d);
  cudaDeviceSynchronize();
  vectornorm<<<64,1024>>>(thrust::raw_pointer_cast(frep_exact.data()), thrust::raw_pointer_cast(normas.data()), n, d);
  cudaDeviceSynchronize();
  distance<<<Blocks,threads>>>(thrust::raw_pointer_cast(frep_exact.data()),frep,n,d);
 cudaDeviceSynchronize();
  dataPoint error = thrust::reduce(frep_exact.begin(), frep_exact.begin()+n,0.0);
  dataPoint norm =thrust::reduce(normas.begin(), normas.begin()+n,0.0);
  error=std::sqrt(error)/std::sqrt(norm);

std::cout<<"Exit_Compute_Error "<<error <<"\n";
  return error;

}
