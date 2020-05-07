#ifndef COMMON
#define COMMON
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <iostream>
#include <string.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <stdint.h>
typedef double coord;
typedef double matval;   //!< Data-type of sparse matrix elements
typedef uint32_t matidx; //!< Data-type of sparse matrix indices
#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }


#endif
