
#ifndef PERPLEXITY_SEARCH_CUH
#define PERPLEXITY_SEARCH_CUH
#include "types.hpp"
#include "common.cuh"
#include "utils_cuda.cuh"
#include <assert.h>
#include <cublas_v2.h>
#include <float.h>
#include <iostream>
#include <limits>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#include <thrust/transform.h>
template <class dataPoint>
sparse_matrix<dataPoint> perplexityEqualization(int *I, dataPoint *D, int n,
                                                int nn, dataPoint u);


#endif
