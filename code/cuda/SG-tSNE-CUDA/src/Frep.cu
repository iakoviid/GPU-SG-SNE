
#include "Frep.cuh"
#include "timerstream.h"
#include "complex.cuh"
#include "complexD.cuh"


#define num_threads 1024
extern cudaStream_t streamRep;

template <class dataPoint>
__global__ void ComputeChargesKernel(volatile dataPoint *__restrict__ VScat,
                                     const dataPoint *const y, const int n,
                                     const int d) {
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n) {
    return;
  }
  switch (d) {
  case 1: {
    VScat[TID] = 1;
    VScat[TID + n] = y[TID];
    break;
  }
  case 2: {
    VScat[3*TID] = 1;
    VScat[3*TID + 1] = y[TID];
    VScat[3*TID + 2] = y[TID + n];
    break;
  }
  case 3: {
    VScat[TID] = 1;
    VScat[TID + n] = y[TID];
    VScat[TID + 2 * n] = y[TID + n];
    VScat[TID + 3 * n] = y[TID + 2 * n];
    break;
  }
  }
}
template <class dataPoint>
void ComputeCharges(dataPoint *VScat, dataPoint *y_d, const int n,
                    const int d) {
  const int num_blocks = (n + num_threads - 1) / num_threads;
  ComputeChargesKernel<<<num_blocks, num_threads, 0, streamRep>>>(VScat, y_d, n,
                                                                  d);
}
template <class dataPoint>
__global__ void
compute_repulsive_forces_kernel(volatile dataPoint *__restrict__ frep,
                                const dataPoint *const Y, const int num_points,
                                const int nDim, const dataPoint *const Phi,
                                volatile dataPoint *__restrict__ zetaVec, uint32_t *iPerm) {

  register dataPoint Ysq = 0;
  register dataPoint z = 0;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < num_points; TID += gridDim.x * blockDim.x) {
    Ysq = 0;
    z = 0;
    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[TID + j * num_points] * Y[TID + j * num_points];
      z -= 2 * Y[TID + j * num_points] * Phi[TID + (num_points) * (j + 1)];
      frep[iPerm[TID] + j * num_points] =
          Y[TID + j * num_points] * Phi[TID] - Phi[TID + (j + 1) * num_points];
    }

    z += (1 + 2 * Ysq) * Phi[TID];
    zetaVec[TID] = z;
  }
}

template <class dataPoint>
dataPoint zetaAndForce(dataPoint *Ft_d, dataPoint *y_d, int n, int d,
                       dataPoint *Phi,uint32_t *iPerm,thrust::device_vector<dataPoint> &zetaVec) {
  // can posibly reduce amongs the threads and then divide

  int threads = 1024;
  int Blocks = 64;
  compute_repulsive_forces_kernel<<<Blocks, threads, 0, streamRep>>>(
      Ft_d, y_d, n, d, Phi, thrust::raw_pointer_cast(zetaVec.data()),iPerm);
      cudaDeviceSynchronize();
  dataPoint z = thrust::reduce(thrust::cuda::par.on(streamRep), zetaVec.begin(),
                               zetaVec.end()) -
                n;

  ArrayScale<<<Blocks, threads, 0, streamRep>>>(Ft_d, 1 / z, n * d);
  return z;
}

template <class dataPoint, class Complext>
dataPoint computeFrepulsive_interp(dataPoint *Frep, dataPoint *y, int n, int d,
                                   dataPoint h, double *timeInfo, int nGrid,
                                   cufftHandle &plan, cufftHandle &plan_rhs,
                                   dataPoint *yt, dataPoint *VScat,
                                   dataPoint *PhiScat, dataPoint *VGrid,
                                   dataPoint *PhiGrid, Complext *Kc,
                                   Complext *Xc,thrust::device_vector<dataPoint> &zetaVec)  {

  struct GpuTimer timer;

  const int num_blocks = (n + num_threads - 1) / num_threads;
  dataPoint* yr;
  CUDA_CALL(cudaMallocManaged(&yr, (d) * n * sizeof(dataPoint)));

  // ~~~~~~~~~~ move data to (0,0,...)
  dataPoint miny[4];


  timer.Start(streamRep);
  thrust::device_ptr<dataPoint> yVec_ptr(y);
  for (int j = 0; j < d; j++) {
    miny[j] = thrust::reduce(yVec_ptr+j*n, yVec_ptr+n*(j+1),100.0, thrust::minimum<dataPoint>());
    addScalar<<<num_blocks, num_threads>>>(&y[j * n], -miny[j], n);
  }

  timer.Stop(streamRep);
  timeInfo[6] = timer.Elapsed();

  ArrayCopy<<<64, num_threads, 0, streamRep>>>(y, yt, n * d);
  int *ib;
  uint32_t points= pow(nGrid-1, d)+1;
  CUDA_CALL(cudaMallocManaged(&ib, points* sizeof(int)));
  thrust::device_vector<uint32_t> iPerm(n);
  thrust::sequence(iPerm.begin(), iPerm.end());
  timer.Start(streamRep);

  relocateCoarseGrid(yt,thrust::raw_pointer_cast(iPerm.data()),ib,n,nGrid,d);

  timer.Stop(streamRep);
  timeInfo[0]= timer.Elapsed();

  ComputeCharges(VScat, yt, n, d);
  ArrayCopy<<<32,256>>>(yt,yr,n*d);
  timer.Start(streamRep);
  nuconv(PhiScat, yt, VScat, ib, n, d, d + 1, nGrid, timeInfo + 1, plan,
         plan_rhs,VGrid,PhiGrid,Kc,Xc);
  timer.Stop(streamRep);
  timeInfo[5] = timer.Elapsed();
  timer.Start(streamRep);
  dataPoint zeta = zetaAndForce(Frep, yr, n, d, PhiScat, thrust::raw_pointer_cast(iPerm.data()),zetaVec);
  timer.Stop(streamRep);
  timeInfo[4] = timer.Elapsed();
  CUDA_CALL(cudaFree(yr));
  CUDA_CALL(cudaFree(ib));

  return zeta;
}
template float computeFrepulsive_interp(
    float *Frep, float *y, int n, int d, float h, double *timeInfo, int nGrid,
    cufftHandle &plan, cufftHandle &plan_rhs, float *yt, float *VScat,
    float *PhiScat, float *VGrid, float *PhiGrid, Complex *Kc, Complex *Xc,thrust::device_vector<float> &zetaVecs);
template double computeFrepulsive_interp(double *Frep, double *y, int n, int d,
                                         double h, double *timeInfo, int nGrid,
                                         cufftHandle &plan,
                                         cufftHandle &plan_rhs, double *yt,
                                         double *VScat, double *PhiScat,
                                         double *VGrid, double *PhiGrid,
                                         ComplexD *Kc, ComplexD *Xc,thrust::device_vector<double> &zetaVecs);
