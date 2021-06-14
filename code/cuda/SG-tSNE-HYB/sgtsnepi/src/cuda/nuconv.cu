
#include "nuconv.cuh"
#include "gpu_timer.h"
#include "utils_cuda.cuh"
#include "complexF.cuh"
#include "complexD.cuh"
#include <fstream>
#include <float.h>
extern cudaStream_t streamRep;
#define Blocks 64
#define Threads 1024

__global__ void Normalize(volatile float *__restrict__ y,
                          const uint32_t nPts, const uint32_t ng,
                          const uint32_t d, const float maxy) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (register int dim = 0; dim < d; dim++) {
      y[TID + dim * nPts] /= maxy;
      if (y[TID + dim * nPts] >= 1) {
        y[TID + dim * nPts] =1 - FLT_EPSILON;
      }
      y[TID + dim * nPts] *= (ng - 3);
    }
  }
}
__global__ void Normalize(volatile double *__restrict__ y,
                          const uint32_t nPts, const uint32_t ng,
                          const uint32_t d, const double maxy) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (register int dim = 0; dim < d; dim++) {
      y[TID + dim * nPts] /= maxy;
      if (y[TID + dim * nPts] == 1) {
        y[TID + dim * nPts] = 1 - DBL_EPSILON;
      }
      y[TID + dim * nPts] *= (ng - 3);
    }
  }
}
template <class dataPoint,class Complext>
void nuconv(dataPoint *PhiScat, dataPoint *y, dataPoint *VScat,  int n,
            int d, int m, int nGridDim, double *timeInfo, cufftHandle &plan,
            cufftHandle &plan_rhs, dataPoint *VGrid,
            dataPoint *PhiGrid, Complext *Kc,
            Complext *Xc) {
  struct GpuTimer timer;
  int szV = pow(nGridDim + 2, d) * m;
  timer.Start(streamRep);

 // ~~~~~~~~~~ Scale coordinates (inside bins)
  thrust::device_ptr<dataPoint> yVec_ptr(y);
  dataPoint maxy =
      thrust::reduce(thrust::cuda::par.on(streamRep), yVec_ptr,
                     yVec_ptr + n * d, 0.0, thrust::maximum<dataPoint>());
 //cudaDeviceSynchronize();


  dataPoint h =
      maxy / (nGridDim - 1 - std::numeric_limits<dataPoint>::epsilon());

  // ~~~~~~~~~~ scat2grid

  Normalize<<<Blocks, Threads, 0, streamRep>>>(y, n, nGridDim + 2, d, maxy);
  //cudaDeviceSynchronize();

 timer.Stop(streamRep);

  timeInfo[5] += timer.Elapsed()/1000.0;

  timer.Start(streamRep);
  s2g(VGrid, y, VScat, nGridDim, n, d, m);

  timer.Stop(streamRep);
  timeInfo[0] += timer.Elapsed()/1000.0;

 //cudaDeviceSynchronize();



 // ~~~~~~~~~~ grid2grid

  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = nGridDim + 2;
  }
  timer.Start(streamRep);

  switch (d) {

  case 1:
    conv1dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d, plan, plan_rhs,Kc,Xc);

    break;

  case 2:
    conv2dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d, plan, plan_rhs,Kc,Xc);

    break;

  case 3:
    conv3dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d, plan, plan_rhs,Kc,Xc);
    break;
  }
// cudaDeviceSynchronize();
  timer.Stop(streamRep);
  timeInfo[1] = timer.Elapsed()/1000.0;

  // ~~~~~~~~~~ grid2scat
  timer.Start(streamRep);
  g2s(PhiScat, PhiGrid, y, nGridDim, n, d, m);

  timer.Stop(streamRep);
  timeInfo[2] = timer.Elapsed()/1000.0;
  // ~~~~~~~~~~ deallocate memory
 //cudaDeviceSynchronize();
  delete nGridDims;
}
template void nuconv(float *PhiScat, float *y, float *VScat,  int n,
                     int d, int m, int nGridDim, double *timeInfo,
                     cufftHandle &plan, cufftHandle &plan_rhs, float *VGrid, float *PhiGrid, ComplexF *Kc, ComplexF *Xc);
template void nuconv(double *PhiScat, double *y, double *VScat,  int n,
                     int d, int m, int nGridDim, double *timeInfo,
                     cufftHandle &plan, cufftHandle &plan_rhs, double *VGrid, double *PhiGrid,
                      ComplexD *Kc, ComplexD *Xc);
