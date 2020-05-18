#ifndef UTILS_CUH
#define UTILS_CUH
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

#endif
