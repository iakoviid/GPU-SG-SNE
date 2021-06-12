
#include "nuconv.cuh"
#include "timer.h"
#include "utils_cuda.cuh"
#define E_LIMITF 1.19209290e-007
#define E_LIMITD 2.22045e-16
__global__ void Normalize(volatile float *__restrict__ y,
                          const uint32_t nPts, const uint32_t ng,
                          const uint32_t d, const float maxy) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (register int dim = 0; dim < d; dim++) {
      y[TID + dim * nPts] /= maxy;
      if (y[TID + dim * nPts] == 1) {
        y[TID + dim * nPts] = y[TID + dim * nPts] - E_LIMITF;
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
        y[TID + dim * nPts] = y[TID + dim * nPts] - E_LIMITD;
      }
      y[TID + dim * nPts] *= (ng - 3);
    }
  }
}
template <class dataPoint>
void nuconv(dataPoint *PhiScat, dataPoint *y, dataPoint *VScat, int *ib, int n,
            int d, int m, int nGridDim, double *timeInfo,cufftHandle& plan,cufftHandle& plan_rhs) {
  struct GpuTimer timer;

  // ~~~~~~~~~~ Scale coordinates (inside bins)
  thrust::device_ptr<dataPoint> yVec_ptr(y);
  dataPoint maxy = thrust::reduce(yVec_ptr, yVec_ptr + n * d, 0.0,
                                  thrust::maximum<dataPoint>());

  dataPoint h =
      maxy / (nGridDim - 1 - std::numeric_limits<dataPoint>::epsilon());
  std::cout<"h= "<<h<<"\n";
  // ~~~~~~~~~~ scat2grid
  int szV = pow(nGridDim + 2, d) * m;
  dataPoint *VGrid;
  CUDA_CALL(cudaMallocManaged(&VGrid, szV * sizeof(dataPoint)));
  initKernel<dataPoint><<<64, 1024>>>(VGrid, 0, szV);

  Normalize<<<64, 1024>>>(y, n, nGridDim + 2, d, maxy);
  timer.Start();

  s2g(VGrid, y, VScat, nGridDim, n, d, m, ib);

  timer.Stop();
  timeInfo[0] = timer.Elapsed();
  // ~~~~~~~~~~ grid2grid
  cudaDeviceSynchronize();

  dataPoint *PhiGrid;
  CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(dataPoint)));
  initKernel<dataPoint><<<64, 1024>>>(PhiGrid, 0, szV);

  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = nGridDim + 2;
  }
  timer.Start();

  switch (d) {

  case 1:
    conv1dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d, plan, plan_rhs);

    break;

  case 2:
    conv2dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d,plan,plan_rhs);

    break;

  case 3:
    conv3dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d, plan, plan_rhs);
    break;
  }

  timer.Stop();
  timeInfo[1] = timer.Elapsed();

  // ~~~~~~~~~~ grid2scat
  timer.Start();
  g2s(PhiScat, PhiGrid, y, nGridDim, n, d, m);

  timer.Stop();
  timeInfo[2] = timer.Elapsed();
  // ~~~~~~~~~~ deallocate memory
  CUDA_CALL(cudaFree(PhiGrid));
  CUDA_CALL(cudaFree(VGrid));
  delete nGridDims;
}
template void nuconv(float *PhiScat, float *y, float *VScat, int *ib, int n,
                     int d, int m, int nGridDim, double *timeInfo,cufftHandle& plan,cufftHandle& plan_rhs);
template void nuconv(double *PhiScat, double *y, double *VScat, int *ib, int n, int d,
            int m, int nGridDim, double *timeInfo,cufftHandle& plan,cufftHandle& plan_rhs);
