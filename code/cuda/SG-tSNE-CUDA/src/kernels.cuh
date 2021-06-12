#include <cuda_runtime.h>
#ifndef KERNELS
#define KERNELS
__inline__ __device__ __host__ coord kernel1d(coord hsq, coord i) {
  return pow(1.0 + hsq * i * i, -2);
}

__inline__ __device__ __host__ coord kernel2d(coord hsq, coord i, coord j) {
  return pow(1.0 + hsq * (i * i + j * j), -2);
}

__inline__ __device__ __host__ coord kernel3d(coord hsq, coord i, coord j,
                                              coord k) {
  return pow(1.0 + hsq * (i * i + j * j + k * k), -2);
}
#endif
