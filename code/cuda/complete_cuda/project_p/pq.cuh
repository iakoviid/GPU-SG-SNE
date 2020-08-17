#ifndef PQ_CUH
#define PQ_CUH
#include "types.hpp"
#include "common.cuh"
#include "utils_gpu.cuh"

__global__ void PQKernel(coord *Fattr, coord *const Y, double const *const p_sp,
                         matidx *ir, matidx *jc, int const n, int const d);
#endif
