#include "non_periodic_conv.cuh"
#include "utils_cuda.cuh"
#include "matrix_indexing.hpp"
#define idx2(i,j,d) (SUB2IND2D(i,j,d))
#define idx3(i,j,k,d1,d2) (SUB2IND3D(i,j,k,d1,d2))

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

__global__ void setDataFft1D(Complex *Kc, Complex *Xc, int ng, int nVec,
                             coord *VGrid, coord hsq, int sign) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Kc[i].x = kernel1d(hsq, i);
    if (i > 0) {
      Kc[i].x = Kc[i].x + sign * kernel1d(hsq, ng - i);
      if (sign == -1) {

        Complex arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2*ng);
        Kc[i] = ComplexMul(Kc[i], my_cexpf(arg));
      }
    }
    for (int j = 0; j < nVec; j++) {
      Xc[i + j * ng].x = VGrid[i + j * ng];
      if (sign == -1) {
        Complex arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i /(2* ng);
        Xc[i + j * ng] = ComplexMul(Xc[i + j * ng], my_cexpf(arg));
      }
    }
  }
}

__global__ void setDataFft2D(Complex *Kc, Complex *Xc, int n1, int n2, int nVec,
                             const coord *const VGrid, coord hsq, int signx,int signy) {



  for (uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n2;
       j += blockDim.y * gridDim.y) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n1;
         i += blockDim.x * gridDim.x) {
           Kc[idx2(i,j,n1)].x = kernel2d( hsq, i, j );
           if (i > 0) {Kc[idx2(i,j,n1)].x += signx * kernel2d( hsq,n1-i, j );}
           if (j > 0) {Kc[idx2(i,j,n1)].x +=  signy * kernel2d( hsq,i,n2-j );}
           if (i>0 && j > 0) {Kc[idx2(i,j,n1)].x += signx*signy * kernel2d( hsq,n1-i,n2-j );}

           for(uint32_t iVec = 0; iVec < nVec; iVec++){
             Xc[ idx3(i, j, iVec ,n1, n2) ].x=VGrid[ idx3(i, j, iVec, n1, n2) ];
               if(signx==-1){
                 Complex arg;
                 arg.x = 0;
                 arg.y = -2 * CUDART_PI_F * i / (2*n1);
                 Xc[ idx3(i, j, iVec ,n1, n2) ] = ComplexMul(Xc[ idx3(i, j, iVec ,n1, n2) ], my_cexpf(arg));
               }
               if(signy==-1){
                 Complex arg;
                 arg.x = 0;
                 arg.y = -2 * CUDART_PI_F * j / (2*n2);
                 Xc[ idx3(i, j, iVec ,n1, n2) ] = ComplexMul(Xc[ idx3(i, j, iVec ,n1, n2) ], my_cexpf(arg));
               }

           }
           if(signx==-1){
             Complex arg;
             arg.x = 0;
             arg.y = -2 * CUDART_PI_F * i / (2*n1);
             Kc[idx2(i,j,n1)] = ComplexMul(Kc[idx2(i,j,n1)], my_cexpf(arg));
           }

           if(signy==-1){
             Complex arg;
             arg.x = 0;
             arg.y = -2 * CUDART_PI_F * j / (2*n2);
             Kc[idx2(i,j,n1)] = ComplexMul(Kc[idx2(i,j,n1)], my_cexpf(arg));
           }


    }
  }

}
__global__ void addToPhiGrid(Complex *Xc, coord *PhiGrid, int ng,coord scale) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    PhiGrid[i] += scale* Xc[i].x;
  }
}

__global__ void normalizeInverse(Complex *Xc, int ng) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Complex arg;
    arg.x = 0;
    arg.y = +2 * CUDART_PI_F * i / (2*ng);
    Xc[i] = ComplexMul(Xc[i], my_cexpf(arg));
  }
}

__global__ void normalizeInverse2D(Complex *Xc, uint32_t n1, uint32_t n2,
                                   uint32_t nVec, int signx, int signy) {


  for (uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n2;
       j += blockDim.y * gridDim.y) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n1;
         i += blockDim.x * gridDim.x) {
      for (uint32_t iVec = 0; iVec < nVec; iVec++) {
        if (signx == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = +2 * CUDART_PI_F * i / (2 * n1);
          Xc[idx3(i, j, iVec, n1, n2)] =
              ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
        }
        if (signy == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = +2 * CUDART_PI_F * j / (2 * n2);
          Xc[idx3(i, j, iVec, n1, n2)] =
              ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
        }
      }
    }
  }
}

void conv1dnopadcuda(coord *PhiGrid, coord *VGrid, coord h, int nGridDim,
                     int nVec, int nDim,
                     Complex *Kc,Complex* Xc,cufftHandle plan,cufftHandle plan_rhs

                   ) {

  coord hsq = h * h;
  Complex *Kc, *Xc;
  CUDA_CALL(cudaMallocManaged(&Kc, nGridDim * sizeof(Complex)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * nGridDim * sizeof(Complex)));

  cufftHandle plan;
  cufftPlan1d(&plan, nGridDim, CUFFT_C2C, 1);
  /*even*/
  setDataFft1D<<<32, 256>>>(Kc, Xc, nGridDim, nVec, VGrid, hsq, 1);

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
    addToPhiGrid<<<32, 256>>>(&Xc[j * nGridDim], &PhiGrid[j * nGridDim],
                              nGridDim,(0.5 / nGridDim));
  }

  cudaDeviceSynchronize(); // why

  setDataFft1D<<<64, 1024>>>(Kc, Xc, nGridDim, nVec, VGrid, hsq, -1);

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
    normalizeInverse<<<32, 256>>>(&Xc[j * nGridDim], nGridDim);
    addToPhiGrid<<<32, 256>>>(&Xc[j * nGridDim], &PhiGrid[j * nGridDim],nGridDim,(0.5 / nGridDim));
  }

  return;
}
void conv2dnopadcuda(coord *const PhiGrid, const coord *const VGrid,
                     const coord h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim,

                     Complex *Kc,Complex* Xc,cufftHandle plan,cufftHandle plan_rhs
                    ) {
  coord hsq = h * h;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1 = nGridDims[0];
  uint32_t n2 = nGridDims[1];
  int ng[2]={n1,n2};


  // ============================== EVEN-EVEN

  setDataFft2D<<<32, 256>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, 1, 1);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);

  for (int j = 0; j < nVec; j++) {
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j * n1 * n2], Kc, n1 * n2,
                                             1.0f);
  }

  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  addToPhiGrid<<<32, 256>>>(Xc, PhiGrid,  n1*n2*nVec, (0.25 / (n1 * n2)));


  // ============================== ODD-EVEN

  setDataFft2D<<<32, 256>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, -1, 1);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);

  for (int j = 0; j < nVec; j++) {
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j * n1 * n2], Kc, n1 * n2,
                                             1.0f);
  }
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  normalizeInverse2D<<<32, 256>>>(Xc, n1, n2, nVec, -1, 1);
  addToPhiGrid<<<32, 256>>>(Xc, PhiGrid, n1*n2*nVec, (0.25 / (n1 * n2)));

  // ============================== EVEN-ODD

  setDataFft2D<<<32, 256>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, 1, -1);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);
  for (int j = 0; j < nVec; j++) {
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j * n1 * n2], Kc, n1 * n2,
                                             1.0f);
  }
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);
  normalizeInverse2D<<<32, 256>>>(Xc, n1, n2, nVec, 1, -1);

  addToPhiGrid<<<32, 256>>>(Xc, PhiGrid, n1*n2*nVec, (0.25 / (n1 * n2)));

  // ============================== ODD-ODD

  setDataFft2D<<<32, 256>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, -1, -1);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(Kc),
               reinterpret_cast<cufftComplex *>(Kc), CUFFT_FORWARD);
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_FORWARD);

  for (int j = 0; j < nVec; j++) {
    ComplexPointwiseMulAndScale<<<32, 256>>>(&Xc[j * n1 * n2], Kc, n1 * n2,
                                             1.0f);
  }
  cufftExecC2C(plan_rhs, reinterpret_cast<cufftComplex *>(Xc),
               reinterpret_cast<cufftComplex *>(Xc), CUFFT_INVERSE);

  normalizeInverse2D<<<32, 256>>>(Xc, n1, n2, nVec, -1, -1);
  addToPhiGrid<<<32, 256>>>(Xc, PhiGrid, n1*n2*nVec, (0.25 / (n1 * n2)));


}
