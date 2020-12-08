#include "matrix_indexing.hpp"
#include "non_periodic_convD.cuh"
#include "utils_cuda.cuh"
#include <sys/time.h>

#define idx2(i, j, d) (SUB2IND2D(i, j, d))
#define idx3(i, j, k, d1, d2) (SUB2IND3D(i, j, k, d1, d2))
#define idx4(i, j, k, l, m, n, o) (SUB2IND4D(i, j, k, l, m, n, o))
#define CUDART_PI_F 3.14159265358979323846
#define Blocks 64
#define Threads 512
#define Blocks2D 64
#define Threads2D 512
#define Blocks3D 64
#define Threads3D 512
__host__ __device__ int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

// Complex pointwise multiplication
static __global__ void ComplexPointwiseMulAndScale(Complex *a, const Complex *b,
                                                   int size, uint32_t nVec) {
  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int j = 0; j < nVec; j++) {

    for (int i = threadID; i < size; i += numThreads) {
      a[i + j * size] = ComplexScale(ComplexMul(a[i + j * size], b[i]), 1.0f);
    }
  }
}

__global__ void setDataFft1D(Complex *Kc, Complex *Xc, int ng, int nVec,
                             coord *VGrid, coord hsq, int sign) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    Kc[i].x = kernel1d(hsq, i);
    Kc[i].y=0;
    if (i > 0) {
      Kc[i].x = Kc[i].x + sign * kernel1d(hsq, ng - i);
      if (sign == -1) {

        Complex arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * ng);
        Kc[i] = ComplexMul(Kc[i], my_cexpf(arg));
      }
    }
    for (int j = 0; j < nVec; j++) {
      Xc[i + j * ng].x = VGrid[i + j * ng];
      Xc[i + j * ng].y=0;
      if (sign == -1) {
        Complex arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * ng);
        Xc[i + j * ng] = ComplexMul(Xc[i + j * ng], my_cexpf(arg));
      }
    }
  }
}

__global__ void setDataFft2D(Complex *Kc, Complex *Xc, int n1, int n2, int nVec,
                             const coord *const VGrid, coord hsq, int signx,
                             int signy) {

  for (uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n2;
       j += blockDim.y * gridDim.y) {
    for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n1;
         i += blockDim.x * gridDim.x) {
      Kc[idx2(i, j, n1)].x = kernel2d(hsq, i, j);
      Kc[idx2(i, j, n1)].y=0;
      if (i > 0) {
        Kc[idx2(i, j, n1)].x += signx * kernel2d(hsq, n1 - i, j);
      }
      if (j > 0) {
        Kc[idx2(i, j, n1)].x += signy * kernel2d(hsq, i, n2 - j);
      }
      if (i > 0 && j > 0) {
        Kc[idx2(i, j, n1)].x += signx * signy * kernel2d(hsq, n1 - i, n2 - j);
      }

      for (uint32_t iVec = 0; iVec < nVec; iVec++) {
        Xc[idx3(i, j, iVec, n1, n2)].x = VGrid[idx3(i, j, iVec, n1, n2)];
        Xc[idx3(i, j, iVec, n1, n2)].y=0;
        if (signx == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = -2 * CUDART_PI_F * i / (2 * n1);
          Xc[idx3(i, j, iVec, n1, n2)] =
              ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
        }
        if (signy == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = -2 * CUDART_PI_F * j / (2 * n2);
          Xc[idx3(i, j, iVec, n1, n2)] =
              ComplexMul(Xc[idx3(i, j, iVec, n1, n2)], my_cexpf(arg));
        }
      }
      if (signx == -1) {
        Complex arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * i / (2 * n1);
        Kc[idx2(i, j, n1)] = ComplexMul(Kc[idx2(i, j, n1)], my_cexpf(arg));
      }

      if (signy == -1) {
        Complex arg;
        arg.x = 0;
        arg.y = -2 * CUDART_PI_F * j / (2 * n2);
        Kc[idx2(i, j, n1)] = ComplexMul(Kc[idx2(i, j, n1)], my_cexpf(arg));
      }
    }
  }
}
__global__ void addToPhiGrid(Complex *Xc, coord *PhiGrid, int ng, coord scale) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    PhiGrid[i] += scale * Xc[i].x;
  }
}

__global__ void normalizeInverse(Complex *Xc, int ng, uint32_t nVec) {

  const int numThreads = blockDim.x * gridDim.x;
  const int threadID = blockIdx.x * blockDim.x + threadIdx.x;
  for (int i = threadID; i < ng; i += numThreads) {
    for (uint32_t iVec = 0; iVec < nVec; iVec++) {
      Complex arg;
      arg.x = 0;
      arg.y = +2 * CUDART_PI_F * i / (2 * ng);
      Xc[i + iVec * ng] = ComplexMul(Xc[i + iVec * ng], my_cexpf(arg));
    }
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

void conv1dnopadcuda(coord *PhiGrid, coord *VGrid, coord h,
                     uint32_t *const nGridDims, uint32_t nVec, int nDim)
{

  uint32_t n1 = nGridDims[0];
  coord hsq = h * h;
  Complex *Kc, *Xc;
  CUDA_CALL(cudaMallocManaged(&Kc, n1 * sizeof(Complex)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * n1 * sizeof(Complex)));

  cufftHandle plan, plan_rhs;
  int ng[1] = {(int)n1};

  cufftPlan1d(&plan, n1, CUFFT_Z2Z, 1);
  cufftPlanMany(&plan_rhs, 1, ng, NULL, 1, n1, NULL, 1, n1, CUFFT_Z2Z, nVec);
  //struct timeval t1, t2;
  //double elapsedTime;
  //gettimeofday(&t1, NULL);

  /*even*/
  setDataFft1D<<<Blocks, Threads>>>(Kc, Xc, n1, nVec, VGrid, hsq, 1);

  cudaDeviceSynchronize(); // why

  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc, n1, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);
  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * nVec, (0.5 / n1));

  cudaDeviceSynchronize(); // why

  setDataFft1D<<<Blocks, Threads>>>(Kc, Xc, n1, nVec, VGrid, hsq, -1);

  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);


  ComplexPointwiseMulAndScale<<<Blocks, Threads>>>(Xc, Kc, n1, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);

  normalizeInverse<<<Blocks, Threads>>>(Xc, n1, nVec);

  addToPhiGrid<<<Blocks, Threads>>>(Xc, PhiGrid, n1 * nVec, (0.5 / n1));

  //cudaDeviceSynchronize();
  //gettimeofday(&t2, NULL);

  //elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  //elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  //printf("CUDA nopad time %lf\n", elapsedTime);

  CUDA_CALL(cudaFree(Kc));
  CUDA_CALL(cudaFree(Xc));
  return;
}
void conv2dnopadcuda(coord *const PhiGrid, const coord *const VGrid,
                     const coord h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim)
{
  coord hsq = h * h;
  Complex *Kc, *Xc;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1 = nGridDims[0];
  uint32_t n2 = nGridDims[1];
  dim3 block(32, 32);
  dim3 grid(iDivUp(n1,32),iDivUp(n2,32));
  //int grid=64;
  //int block=256;
  int ng[2] = {(int)n1, (int)n2};
  CUDA_CALL(cudaMallocManaged(&Kc, n1 * n2 * sizeof(Complex)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * n1 * n2 * sizeof(Complex)));
  cufftHandle plan, plan_rhs;
  cufftPlanMany(&plan, 2, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
  cufftPlanMany(&plan_rhs, 2, ng, NULL, 1, n1 * n2, NULL, 1, n1 * n2, CUFFT_Z2Z,
                nVec);
  //struct timeval t1, t2;
  //double elapsedTime;
  //gettimeofday(&t1, NULL);
  // ============================== EVEN-EVEN

  setDataFft2D<<<grid, block>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, 1, 1);
  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks2D, Threads2D>>>(Xc, Kc, n1 * n2, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);
  addToPhiGrid<<<Blocks2D, Threads2D>>>(Xc, PhiGrid, n1 * n2 * nVec, (0.25 / (n1 * n2)));

  // ============================== ODD-EVEN

  setDataFft2D<<<grid, block>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, -1, 1);
  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks2D, Threads2D>>>(Xc, Kc, n1 * n2, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);
  normalizeInverse2D<<<grid, block>>>(Xc, n1, n2, nVec, -1, 1);
  addToPhiGrid<<<Blocks2D, Threads2D>>>(Xc, PhiGrid, n1 * n2 * nVec, (0.25 / (n1 * n2)));

  // ============================== EVEN-ODD

  setDataFft2D<<<grid, block>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, 1, -1);
  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);
  ComplexPointwiseMulAndScale<<<Blocks2D, Threads2D>>>(Xc, Kc, n1 * n2, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);
  normalizeInverse2D<<<grid, block>>>(Xc, n1, n2, nVec, 1, -1);

  addToPhiGrid<<<Blocks2D, Threads2D>>>(Xc, PhiGrid, n1 * n2 * nVec, (0.25 / (n1 * n2)));

  // ============================== ODD-ODD

  setDataFft2D<<<grid, block>>>(Kc, Xc, n1, n2, nVec, VGrid, hsq, -1, -1);
  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks2D, Threads2D>>>(Xc, Kc, n1 * n2, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);

  normalizeInverse2D<<<grid, block>>>(Xc, n1, n2, nVec, -1, -1);
  addToPhiGrid<<<Blocks2D, Threads2D>>>(Xc, PhiGrid, n1 * n2 * nVec, (0.25 / (n1 * n2)));
  //cudaDeviceSynchronize();
  //gettimeofday(&t2, NULL);

  //elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  //elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  //printf("CUDA nopad time %lf\n", elapsedTime);
  //cudaFree(Kc);
  //cudaFree(Xc);

    CUDA_CALL(cudaFree(Kc));
    CUDA_CALL(cudaFree(Xc));
  return;
}

__global__ void setDataFft3D(Complex *Kc, Complex *Xc, int n1, int n2, int n3,
                             int nVec, const coord *const VGrid, coord hsq,
                             int signx, int signy, int signz) {
  for (uint32_t k = blockIdx.z * blockDim.z + threadIdx.z; k < n3;
       k += blockDim.z * gridDim.z) {

    for (uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n2;
         j += blockDim.y * gridDim.y) {
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n1;
           i += blockDim.x * gridDim.x) {
        Kc[idx3(i, j, k, n1, n2)].x = kernel3d(hsq, i, j, k);
        Kc[idx3(i, j, k, n1, n2)].y=0;
        if (i > 0) {
          Kc[idx3(i, j, k, n1, n2)].x += signx * kernel3d(hsq, n1 - i, j, k);
        }
        if (j > 0) {
          Kc[idx3(i, j, k, n1, n2)].x += signy * kernel3d(hsq, i, n2 - j, k);
        }
        if (i > 0 && j > 0) {
          Kc[idx3(i, j, k, n1, n2)].x +=
              signx*signy * kernel3d(hsq, n1 - i, n2 - j, k);
        }
        if (k > 0) {
          Kc[idx3(i, j, k, n1, n2)].x += signz * kernel3d(hsq, i, j, n3 - k);
        }
        if (k > 0 && i > 0) {
          Kc[idx3(i, j, k, n1, n2)].x +=
              signx*signz * kernel3d(hsq, n1 - i, j, n3 - k);
        }
        if (k > 0 && j > 0) {
          Kc[idx3(i, j, k, n1, n2)].x +=
              signy*signz * kernel3d(hsq, i, n2 - j, n3 - k);
        }
        if (k > 0 && i > 0 && j > 0) {
          Kc[idx3(i, j, k, n1, n2)].x +=
              signx*signy*signz * kernel3d(hsq, n1 - i, n2 - j, n3 - k);
        }

        for (uint32_t iVec = 0; iVec < nVec; iVec++) {
          Xc[idx4(i, j, k, iVec, n1, n2, n3)].x =
              VGrid[idx4(i, j, k, iVec, n1, n2, n3)];
          Xc[idx4(i, j, k, iVec, n1, n2, n3)].y=0;
          if (signx == -1) {
            Complex arg;
            arg.x = 0;
            arg.y = -2 * CUDART_PI_F * i / (2 * n1);
            Xc[idx4(i, j, k, iVec, n1, n2, n3)] =ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
          }
          if (signy == -1) {
            Complex arg;
            arg.x = 0;
            arg.y = -2 * CUDART_PI_F * j / (2 * n2);
            Xc[idx4(i, j, k, iVec, n1, n2, n3)] =ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
          }
          if (signz == -1) {
            Complex arg;
            arg.x = 0;
            arg.y = -2 * CUDART_PI_F * k / (2 * n3);
            Xc[idx4(i, j, k, iVec, n1, n2, n3)] =ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
          }
        }
        if (signx == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = -2 * CUDART_PI_F * i / (2 * n1);
          Kc[idx3(i, j, k, n1, n2)] =
              ComplexMul(Kc[idx3(i, j, k, n1, n2)], my_cexpf(arg));
        }

        if (signy == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = -2 * CUDART_PI_F * j / (2 * n2);
          Kc[idx3(i, j, k, n1, n2)] =
              ComplexMul(Kc[idx3(i, j, k, n1, n2)], my_cexpf(arg));
        }

        if (signz == -1) {
          Complex arg;
          arg.x = 0;
          arg.y = -2 * CUDART_PI_F * k / (2 * n3);
          Kc[idx3(i, j, k, n1, n2)] =
              ComplexMul(Kc[idx3(i, j, k, n1, n2)], my_cexpf(arg));
        }
      }
    }
  }
}

__global__ void normalizeInverse3D(Complex *Xc, uint32_t n1, uint32_t n2,
                                   uint32_t n3, uint32_t nVec, int signx,
                                   int signy, int signz) {

  for (uint32_t k = blockIdx.z * blockDim.z + threadIdx.z; k < n3;
       k += blockDim.z * gridDim.z) {

    for (uint32_t j = blockIdx.y * blockDim.y + threadIdx.y; j < n2;
         j += blockDim.y * gridDim.y) {
      for (uint32_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n1;
           i += blockDim.x * gridDim.x) {
        for (uint32_t iVec = 0; iVec < nVec; iVec++) {
          if (signx == -1) {
            Complex arg;
            arg.x = 0;
            arg.y = +2 * CUDART_PI_F * i / (2 * n1);
            Xc[idx4(i, j, k, iVec, n1, n2, n3)] =
                ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
          }
          if (signy == -1) {
            Complex arg;
            arg.x = 0;
            arg.y = +2 * CUDART_PI_F * j / (2 * n2);
            Xc[idx4(i, j, k, iVec, n1, n2, n3)] =
                ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
          }
          if (signz == -1) {
            Complex arg;
            arg.x = 0;
            arg.y = +2 * CUDART_PI_F * k / (2 * n3);
            Xc[idx4(i, j, k, iVec, n1, n2, n3)] =
                ComplexMul(Xc[idx4(i, j, k, iVec, n1, n2, n3)], my_cexpf(arg));
          }
        }
      }
    }
  }
}
void term3D(Complex *Kc, Complex *Xc, uint32_t n1, uint32_t n2, uint32_t n3,
            uint32_t nVec, const coord *const VGrid, coord *PhiGrid, coord hsq,
            cufftHandle plan, cufftHandle plan_rhs, int signx, int signy,
            int signz) {
              dim3 block(16, 14, 2);
              dim3 grid( iDivUp(n1,16),iDivUp(n2,14),iDivUp(n3,2));

  setDataFft3D<<<grid, block>>>(Kc, Xc, n1, n2, n3, nVec, VGrid, hsq, signx, signy,
                            signz);


  cufftExecZ2Z(plan, reinterpret_cast<cufftDoubleComplex *>(Kc),
               reinterpret_cast<cufftDoubleComplex *>(Kc), CUFFT_FORWARD);
  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_FORWARD);

  ComplexPointwiseMulAndScale<<<Blocks3D, Threads3D>>>(Xc, Kc, n1 * n2*n3, nVec);

  cufftExecZ2Z(plan_rhs, reinterpret_cast<cufftDoubleComplex *>(Xc),
               reinterpret_cast<cufftDoubleComplex *>(Xc), CUFFT_INVERSE);
  if(signx==-1 || signy==-1 || signz==-1){
  normalizeInverse3D<<<grid, block>>>(Xc, n1, n2, n3, nVec, signx, signy, signz);}
  addToPhiGrid<<<Blocks3D, Threads3D>>>(Xc, PhiGrid, n1 * n2 * n3 * nVec,
                            (0.125 / (n1 * n2 * n3)));
}

void conv3dnopadcuda(coord *const PhiGrid, const coord *const VGrid,
                     const coord h, uint32_t *const nGridDims,
                     const uint32_t nVec, const uint32_t nDim) {

  coord hsq = h * h;
  Complex *Kc, *Xc;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1 = nGridDims[0];
  uint32_t n2 = nGridDims[1];
  uint32_t n3 = nGridDims[2];
  int ng[3] = {(int)n1, (int)n2, (int)n3};
  CUDA_CALL(cudaMallocManaged(&Kc, n1 * n2 * n3 * sizeof(Complex)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * n1 * n2 * n3 * sizeof(Complex)));
  cufftHandle plan, plan_rhs;
  cufftPlanMany(&plan, 3, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
  cufftPlanMany(&plan_rhs, 3, ng, NULL, 1, n1 * n2 * n3, NULL, 1, n1 * n2 * n3,
                CUFFT_Z2Z, nVec);

  // ============================== EVEN-EVEN-EVEN


  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, 1,1);


  // ============================== ODD-EVEN-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, 1,
         1);

  // ============================== EVEN-ODD-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, -1,
         1);

  // ============================== ODD-ODD-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, -1,
         1);

  // ============================== EVEN-EVEN-ODD

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, 1,
         -1);

  // ============================== EVEN-ODD-EVEN

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, 1,
         -1);

  // ============================== EVEN-ODD-ODD

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, 1, -1,
         -1);

  // ============================== ODD-ODD-ODD

  term3D(Kc, Xc, n1, n2, n3, nVec, VGrid, PhiGrid, hsq, plan, plan_rhs, -1, -1,
         -1);
  CUDA_CALL(cudaFree(Xc));
  CUDA_CALL(cudaFree(Kc));

}
