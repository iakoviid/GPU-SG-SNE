
#include "Frep.cuh"
extern int Blocks;
extern int Threads;
extern cudaStream_t streamRep;

template <class dataPoint>
__global__ void ComputeChargesKernel(volatile dataPoint *__restrict__ VScat,
                                     const dataPoint *const y, const int n,
                                     const int d) {
    for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < n; TID += gridDim.x * blockDim.x) {

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
    VScat[4*TID] = 1;
    VScat[4*TID + 1] = y[TID];
    VScat[4*TID + 2 ] = y[TID + n];
    VScat[4*TID + 3 ] = y[TID + 2 * n];
    break;
  }
  }}
}

template <class dataPoint>
void ComputeCharges(dataPoint *VScat, dataPoint *y_d, const int n,
                    const int d) {
  ComputeChargesKernel<<<Blocks, Threads, 0, streamRep>>>(VScat, y_d, n,
                                                                  d);
}
template <class dataPoint>
__global__ void
compute_repulsive_forces_kernel(volatile dataPoint *__restrict__ frep,
                                const dataPoint *const Y, const int num_points,
                                const int nDim, const dataPoint *const Phi,
                                volatile dataPoint *__restrict__ zetaVec) {

  register dataPoint Ysq = 0;
  register dataPoint z = 0;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < num_points; TID += gridDim.x * blockDim.x) {
    Ysq = 0;
    z = 0;
    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[TID + j * num_points] * Y[TID + j * num_points];
      z -= 2 * Y[TID + j * num_points] * Phi[TID + (num_points) * (j + 1)];
      frep[TID + j * num_points] =
          Y[TID + j * num_points] * Phi[TID] - Phi[TID + (j + 1) * num_points];
    }

    z += (1 + 2 * Ysq) * Phi[TID];
    zetaVec[TID] = z;
  }
}

template <class dataPoint>
dataPoint zetaAndForce(dataPoint *Ft_d, dataPoint *y_d, int n, int d,
                       dataPoint *Phi,thrust::device_vector<dataPoint> &zetaVec) {
  compute_repulsive_forces_kernel<<<Blocks, Threads, 0, streamRep>>>(
      Ft_d, y_d, n, d, Phi, thrust::raw_pointer_cast(zetaVec.data()));
  dataPoint z = thrust::reduce(thrust::cuda::par.on(streamRep), zetaVec.begin(),
                               zetaVec.end()) -
                n;
  ArrayScale<<<Blocks, Threads, 0, streamRep>>>(Ft_d, 1 / z, n * d);
  return z;
}
template <class dataPoint, class Complext>
dataPoint computeFrepulsive_interp(dataPoint *Frep, dataPoint *y, int n, int d,
                                   dataPoint h, double *timeInfo, int nGrid,
                                   cufftHandle &plan, cufftHandle &plan_rhs,
                                   dataPoint *yt, dataPoint *VScat,
                                   dataPoint *PhiScat, dataPoint *VGrid,
                                   dataPoint *PhiGrid, Complext *Kc,
                                   Complext *Xc,thrust::device_vector<dataPoint> &zetaVec) {

  struct GpuTimer timer;

  // ~~~~~~~~~~ move data to (0,0,...)
  timer.Start(streamRep);
  ArrayCopy<<<Blocks, Threads, 0, streamRep>>>(y, yt, n * d);
  ComputeCharges(VScat, y, n, d);
  timer.Stop(streamRep);
  timeInfo[6] = timer.Elapsed();

  timer.Start(streamRep);
  nuconv(PhiScat, yt, VScat, n, d, d + 1, nGrid, timeInfo + 1, plan,
         plan_rhs,VGrid,PhiGrid,Kc,Xc);
  timer.Stop(streamRep);
  timeInfo[5] = timer.Elapsed();
  timer.Start(streamRep);
  dataPoint zeta = zetaAndForce(Frep, y, n, d, PhiScat,zetaVec);
  timer.Stop(streamRep);
  timeInfo[4] = timer.Elapsed();

  return zeta;
}
template float computeFrepulsive_interp(
    float *Frep, float *y, int n, int d, float h, double *timeInfo, int nGrid,
    cufftHandle &plan, cufftHandle &plan_rhs, float *yt, float *VScat,
    float *PhiScat, float *VGrid, float *PhiGrid, ComplexF *Kc, ComplexF *Xc,thrust::device_vector<float> &zetaVecs);
template double computeFrepulsive_interp(double *Frep, double *y, int n, int d,
                                         double h, double *timeInfo, int nGrid,
                                         cufftHandle &plan,
                                         cufftHandle &plan_rhs, double *yt,
                                         double *VScat, double *PhiScat,
                                         double *VGrid, double *PhiGrid,
                                         ComplexD *Kc, ComplexD *Xc,thrust::device_vector<double> &zetaVec);