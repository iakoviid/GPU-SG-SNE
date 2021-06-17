#ifndef COMMON
#define COMMON

#include <cuda_fp16.h>
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
#include "types.hpp"
#define gpuErrchk(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != cudaSuccess) {
    fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
            line);
    if (abort)
      exit(code);
  }
}
#endif
