#include "non_periodic_conv.cuh"
#define CUDART_PI_F 3.141592654f
// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, float scale) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;

  for (int i = threadID; i < size; i += numThreads) {
    a[i] = ComplexScale(ComplexMul(a[i], b[i]), scale);
  }
}

__global__ void setDataFft1D(Complex *Kc, Complex *Xc, int ng,
                             int nVec, coord *VGrid, coord hsq, int sign) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Kc[i].x = kernel1d(hsq, i);
    if (i > 0) {
      Kc[i].x = Kc[i].x + sign * kernel1d(hsq, ng - i);
      if (sign == -1) {
        Complex arg;
        arg.x = 1;
        arg.y = -2 * CUDART_PI_F * i / ng;
        Kc[i] = ComplexMul(Kc[i], my_cexpf(arg));
      }
    }
    for (int j = 0; j < nVec; j++) {
      Xc[i + j * ng].x = VGrid[i + j * ng];
      if (sign == -1) {
        Complex arg;
        arg.x = 1;
        arg.y = -2 * CUDART_PI_F * i / ng;
        Xc[i + j * ng] = ComplexMul(Xc[i + j * ng], my_cexpf(arg));
      }
    }
  }
}
__global__ void addToPhiGrid(Complex *Xc, coord *PhiGrid, int ng) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    PhiGrid[i] += (0.5 / ng) * Xc[i].x;
  }
}
__global__ void normalizeInverse(Complex *Xc, int ng) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Complex arg;
    arg.x = 1;
    arg.y = +2 * CUDART_PI_F * i / ng;
    Xc[i] = ComplexMul(Xc[i], my_cexpf(arg));
  }
}

void conv1dnopadcuda(coord *PhiGrid_d, coord *VGrid_d, coord h, int nGridDim,
                     int nVec, int nDim) {

  coord hsq = h * h;
  Complex *Kc, *Xc;
  CUDA_CALL(cudaMallocManaged(&Kc, nGridDim * sizeof(Complex)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * nGridDim * sizeof(Complex)));


  cufftHandle plan;
  cufftPlan1d(&plan, nGridDim, CUFFT_C2C, 1);
  /*even*/
  setDataFft1D<<<64, 1024>>>(Kc, Xc, nGridDim, nVec, VGrid_d, hsq, 1);
  cudaDeviceSynchronize(); // why

  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);


  for (int j = 0; j < nVec; j++) {
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 CUFFT_FORWARD);
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j * nGridDim], Kc, nGridDim,
                                             1.0f);
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 CUFFT_INVERSE);
    addToPhiGrid<<<32, 256>>>(&Xc[j * nGridDim], &PhiGrid_d[j * nGridDim],
                              nGridDim);
  }

  return;

  setDataFft1D<<<64, 1024>>>(Kc, Xc, nGridDim, nVec, VGrid_d, hsq, -1);

  for (int j = 0; j < nVec; j++) {
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 CUFFT_FORWARD);
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j * nGridDim], Kc, nGridDim,
                                             1.0f);
    cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 reinterpret_cast<cufftComplex *>(&Xc[j * nGridDim]),
                 CUFFT_INVERSE);
    normalizeInverse<<<32, 256>>>(&Xc[j * nGridDim], nGridDim);
    addToPhiGrid<<<32, 256>>>(&Xc[j * nGridDim], &PhiGrid_d[j * nGridDim],
                              nGridDim);
  }

}
