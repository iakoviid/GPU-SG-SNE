
#include "nuconv.cuh"
#include "timerstream.h"
#include "utils_cuda.cuh"
#include "complex.cuh"
#include "complexD.cuh"
#include <fstream>
#include <float.h>
extern cudaStream_t streamRep;


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
        y[TID + dim * nPts] = y[TID + dim * nPts] - E_LIMITD;
      }
      y[TID + dim * nPts] *= (ng - 3);
    }
  }
}
template <class dataPoint,class Complext>
void nuconv(dataPoint *PhiScat, dataPoint *y, dataPoint *VScat, int *ib, int n,
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
 cudaDeviceSynchronize();

 timer.Stop(streamRep);

  timeInfo[5] += timer.Elapsed();

  dataPoint h =
      maxy / (nGridDim - 1 - std::numeric_limits<dataPoint>::epsilon());

  // ~~~~~~~~~~ scat2grid

  Normalize<<<64, 1024, 0, streamRep>>>(y, n, nGridDim + 2, d, maxy);
  cudaDeviceSynchronize();

int papa=0;

  timer.Start(streamRep);
  s2g(VGrid, y, VScat, nGridDim, n, d, m, ib);

  timer.Stop(streamRep);
  timeInfo[0] = timer.Elapsed();
 cudaDeviceSynchronize();


if(papa==1){
   
std::ifstream Din;
   Din.open("PhiGrid.txt");
   dataPoint * D=(dataPoint *)malloc(sizeof(dataPoint)*szV);
  for(int i=0;i<szV;i++){
        Din>>D[i];
        }
        Din.close();
  CUDA_CALL(cudaMemcpy(VGrid, D, szV * sizeof(dataPoint),cudaMemcpyHostToDevice));
  
free(D);
}
else if(papa==2){
std::ofstream Din;
   Din.open("PhiGridgp.txt");
   dataPoint * D=(dataPoint *)malloc(sizeof(dataPoint)*szV);
  CUDA_CALL(cudaMemcpy(D, VGrid, szV * sizeof(dataPoint),cudaMemcpyDeviceToHost));

for(int iVec=0;iVec<m;iVec++){
 for(int j=0;j<nGridDim+2;j++){
 for(int i=0;i<nGridDim+2;i++){
        Din<<D[i+j*(nGridDim+2)+iVec*(nGridDim+2)*(nGridDim+2)] <<" "; }
   Din<<"\n";}
Din<<"\n\n\n";
}
        Din.close();  
  
free(D);
std::ofstream yout;
yout.open("Voutgp.txt");
 dataPoint * yh=(dataPoint *)malloc(sizeof(dataPoint)*n*3);

  CUDA_CALL(cudaMemcpy(yh, VScat, 3*n * sizeof(dataPoint),cudaMemcpyDeviceToHost));
//for(int i=0;i<n;i++){yout<<yh[i]<<" "<<yh[i+n]<<"\n"; }
for(int i=0;i<n;i++){yout<<yh[i*3]<<" "<<yh[i*3+1]<<" "<<yh[i*3+2]<<"\n"; }
yout.close();
free(yh);
}

 
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
 cudaDeviceSynchronize();
  timer.Stop(streamRep);
  timeInfo[1] = timer.Elapsed();

  // ~~~~~~~~~~ grid2scat
  timer.Start(streamRep);
  g2s(PhiScat, PhiGrid, y, nGridDim, n, d, m);

  timer.Stop(streamRep);
  timeInfo[2] = timer.Elapsed();
  // ~~~~~~~~~~ deallocate memory
 cudaDeviceSynchronize();
  delete nGridDims;
}
template void nuconv(float *PhiScat, float *y, float *VScat, int *ib, int n,
                     int d, int m, int nGridDim, double *timeInfo,
                     cufftHandle &plan, cufftHandle &plan_rhs, float *VGrid, float *PhiGrid, Complex *Kc, Complex *Xc);
template void nuconv(double *PhiScat, double *y, double *VScat, int *ib, int n,
                     int d, int m, int nGridDim, double *timeInfo,
                     cufftHandle &plan, cufftHandle &plan_rhs, double *VGrid, double *PhiGrid,
                      ComplexD *Kc, ComplexD *Xc);
