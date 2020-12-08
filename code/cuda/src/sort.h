#ifndef SORT_H__
#define SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"
#include <cmath>

void radix_sort(uint64_t* const d_out,
    uint64_t* const d_in,
    unsigned int d_in_len,uint32_t stride,uint32_t nbits,uint32_t* iPerm_out,uint32_t* iPerm_in,double* Y_out,double* Y_in,int d);

#endif
