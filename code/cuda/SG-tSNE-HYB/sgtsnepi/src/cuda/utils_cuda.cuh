#ifndef UTILS_CUDA_CUH
#define UTILS_CUDA_CUH

template <typename T>
__global__ void initKernel(T *devPtr, const T val, const size_t nwords) {
  int tidx = threadIdx.x + blockDim.x * blockIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (; tidx < nwords; tidx += stride)
    devPtr[tidx] = val;
}
template <class dataPoint>
__global__ void addScalar(dataPoint *a, dataPoint scalar, uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] += scalar;
  }
}
template <class dataPoint>
__global__ void ArrayScale(volatile dataPoint *__restrict__ a,
                           const dataPoint scalar, const uint32_t length) {
  for (register int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] *= scalar;
  }
}
template <class dataPoint>
__global__ void ArrayCopy(const dataPoint *const a,
                          volatile dataPoint *__restrict__ b,
                          const uint32_t n) {
  register uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  register uint32_t stride = blockDim.x * gridDim.x;
  for (; tidx < n; tidx += stride) {
    b[tidx] = a[tidx];
  }
}

template <class dataPoint>
__global__ void copydataKernel(dataPoint *a, dataPoint *b, uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] = b[i];
  }
}
template <class dataPoint>
__global__ void normalize(dataPoint *P, dataPoint sum, uint32_t nnz) {
  for (uint32_t i = threadIdx.x + blockIdx.x * blockDim.x; i < nnz;
       i += gridDim.x * blockDim.x) {
    P[i] /= sum;
  }
}
inline __host__ __device__ int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}
template <class dataPoint>
__global__ void addScalarToCoord(dataPoint *y, dataPoint scalar, int n,
                                 int coordinate, int d) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    y[TID * d + coordinate] += scalar;
  }
}
#define FULL_WARP_MASK 0xFFFFFFFF
template <class T> __device__ T warp_reduce(T val) {
  for (int offset = 32 / 2; offset > 0; offset /= 2) {
    val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
  }
  return val;
}
#endif
