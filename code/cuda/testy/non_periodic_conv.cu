
#include <cufft.h>
#include <cufftXt.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include "common.h"
typedef float2 Complex;
#define CUDART_PI_F 3.141592654f
__device__ double kernel1d2(double hsq, double i) {
  return pow(1.0 + hsq * i*i, -2);
}

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
__device__ __forceinline__ Complex my_cexpf (Complex z)

{

    Complex res;

    float t = expf (z.x);

    sincosf (z.y, &res.y, &res.x);

    res.x *= t;

    res.y *= t;

    return res;

}


// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}
__global__ void setDataFft1D(Complex* Kc, Complex* Xc,int ng, Complex* wc,int nVec,double* VGrid,double hsq,int sign)
{

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Kc[i].x=kernel1d2( hsq, i );
    if(i>0){
    Kc[i].x=Kc[i].x +sign*kernel1d2(hsq,ng-i);
    if(sign==-1){
      Complex arg;
      arg.x=1;
      arg.y=-2*CUDART_PI_F*i/ng;
      Kc[i]=ComplexMul(Kc[i],my_cexpf(arg));
    }
  }
    for(int j=0;j<nVec;j++){
      Xc[i+j*ng].x=VGrid[i+j*ng];
      if(sign==-1){
        Complex arg;
        arg.x=1;
        arg.y=-2*CUDART_PI_F*i/ng;
        Xc[i+j*ng]=ComplexMul(Xc[i+j*ng],my_cexpf(arg));
      }
  }
  }


}
__global__ void addToPhiGrid(Complex* Xc,double* PhiGrid,int ng){

    const int numThreads = blockDim.x * gridDim.x;
    const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = threadID; i < ng; i += numThreads) {
      PhiGrid[i]+=(0.5/ng)*Xc[i].x;
    }
}
__global__  void normalizeInverse(Complex* Xc,int ng){

      const int numThreads = blockDim.x * gridDim.x;
      const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
      for (int i = threadID; i < ng; i += numThreads) {
        Complex arg;
        arg.x=1;
        arg.y=+2*CUDART_PI_F*i/ng;
        Xc[i]=ComplexMul(Xc[i],my_cexpf(arg));
      }
}



void conv1dnopadcuda(double* PhiGrid_d, double* VGrid_d,double h,int nGridDim,int nVec,int nDim){

  double hsq=h*h;
  Complex* Kc;
  cudaMallocManaged(&Kc, nGridDim*sizeof(Complex));
  Complex* Xc;
  cudaMallocManaged(&Xc, nVec*nGridDim*sizeof(Complex));
  Complex* wc;
  double*Vmarks=(double *)malloc(sizeof(double)*nVec*nGridDim);
  cudaMemcpy(Vmarks,VGrid_d,sizeof(double)*nVec*nGridDim, cudaMemcpyDeviceToHost);
  for(int i=0;i<nVec*nGridDim;i++){
    //printf("V=%lf\n",Vmarks[i] );
  }
  cufftHandle plan;
  cufftPlan1d(&plan, nGridDim, CUFFT_C2C, 1);
  /*even*/
  setDataFft1D<<<64,1024>>>(Kc,Xc,nGridDim,wc,nVec,VGrid_d,hsq,1);
  cudaDeviceSynchronize();//why
  Complex* Kcd=(Complex *)malloc(sizeof(Complex)*nGridDim);
  Complex* Xch=(Complex *)malloc(sizeof(Complex)*nVec*nGridDim);
  cudaMemcpy(Kcd,Kc,sizeof(Complex)*nGridDim, cudaMemcpyDeviceToHost);
  //printf("========================Preefft=====================================\n" );
  for(int i=0;i<nGridDim;i++){
    //printf("Cuda Kc[%d]=%f +%f i\n",i,Kcd[i].x,Kc[i].y);
  }
  //printf("=============================================================\n" );

  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),reinterpret_cast<cufftComplex *>(Kc),CUFFT_FORWARD);
  cudaMemcpy(Kcd,Kc,sizeof(Complex)*nGridDim, cudaMemcpyDeviceToHost);
  //printf("=========================PostFFT====================================\n" );
  //for(int i=0;i<nGridDim;i++){
    //printf("Cuda Kc[%d]=%f +%f i\n",i,Kcd[i].x,Kc[i].y);
  //}
  printf("========================Preefft=====================================\n" );

  cudaMemcpy(Xch,Xc,sizeof(Complex)*nVec*nGridDim, cudaMemcpyDeviceToHost);
  for(int i=0;i<nVec;i++){
    for(int j=0;j<nGridDim;j++){
    //  printf("Xc[%d]=%f+%f i\n",j+i*nGridDim,Xc[j+i*nGridDim].x,Xc[j+i*nGridDim].y );
    }
  //  printf("\n" );
  }


  for(int j=0;j<nVec;j++){
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),CUFFT_FORWARD);
  ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j*nGridDim], Kc, nGridDim,1.0f );
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),CUFFT_INVERSE);
  addToPhiGrid<<<32,256>>>(&Xc[j*nGridDim],&PhiGrid_d[j*nGridDim],nGridDim);
  }
  printf("=========================PostFFT====================================\n" );


  cudaMemcpy(Xch,Xc,sizeof(Complex)*nVec*nGridDim, cudaMemcpyDeviceToHost);
  for(int i=0;i<nVec;i++){
    for(int j=0;j<nGridDim;j++){
      //printf("Xc[%d]=%f+%f i\n",j+i*nGridDim,Xc[j+i*nGridDim].x,Xc[j+i*nGridDim].y );
    }
    //printf("\n" );
  }
  setDataFft1D<<<64,1024>>>(Kc,Xc,nGridDim,wc,nVec,VGrid_d,hsq,-1);

  for(int j=0;j<nVec;j++){
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),CUFFT_FORWARD);
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j*nGridDim], Kc, nGridDim,1.0f );
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),reinterpret_cast<cufftComplex *>(&Xc[j*nGridDim]),CUFFT_INVERSE);
    normalizeInverse<<<32,256>>>(&Xc[j*nGridDim],nGridDim);
    addToPhiGrid<<<32,256>>>(&Xc[j*nGridDim],&PhiGrid_d[j*nGridDim],nGridDim);
  }
  cudaMemcpy(Vmarks,PhiGrid_d,sizeof(double)*nVec*nGridDim, cudaMemcpyDeviceToHost);
  for(int i=0;i<nVec*nGridDim;i++){
    printf("phi[%d]=%lf\n",i,Vmarks[i] );
  }

  /*odd*/


}
