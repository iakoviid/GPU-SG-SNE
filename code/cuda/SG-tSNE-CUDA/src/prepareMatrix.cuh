
#ifndef SPARSEMATRIX_CUH
#define SPARSEMATRIX_CUH
#include "common.cuh"
#include "matrix_converter.h"
#include <cusparse.h>
#include "utils_cuda.cuh"

template <typename data_type>
sparse_matrix<data_type> *PrepareSparseMatrix(sparse_matrix<data_type> *P,
                                              int *perm, int format,
                                              const char *method, int bs);


void Csr2Coo(sparse_matrix<float> *P);
void Csr2Coo(sparse_matrix<float> **P);
void Csr2Coo(int nnz,int n,int *row, int * col,int* coo_indices);
#endif
