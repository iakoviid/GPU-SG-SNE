
#ifndef SPARSE_REORDER_CUH
#define SPARSE_REORDER_CUH
#include <assert.h>
#include <cusparse.h>
#include "types.hpp"
#include "cusolverSp.h"
#include "helper/helper_cuda.h"
#include <cuda_runtime.h>
template <class dataPoint>
void SparseReorder(const char *method, cusolverSpHandle_t handle,
                  cusparseMatDescr_t descrA, int rowsA, int colsA, int nnzA,
                  int *h_csrRowPtrA, int *h_csrColIndA, dataPoint *h_csrValA,
                  int *h_csrRowPtrB, int *h_csrColIndB, dataPoint *h_csrValB,
                  int *h_Q);
#endif
