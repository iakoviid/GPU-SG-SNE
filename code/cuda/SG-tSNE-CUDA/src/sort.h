#ifndef SORT_H__
#define SORT_H__

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "scan.h"
#include <cmath>
template <class dataPoint>
void radix_sort(uint64_t *const d_out, uint64_t *const d_in,
                unsigned int d_in_len, uint32_t bitStride, uint32_t total,
                uint32_t *iPerm_out, uint32_t *iPerm_in, dataPoint *Y_out,
                dataPoint *Y_in, int d);
#endif
