/*!
  \file   graph_rescaling.hpp
  \brief  Routines implementing lambda-based graph rescaling

*/



#ifndef GRAPH_RESCALING_CUH
#define GRAPH_RESCALING_CUH

#include "types.hpp"
#include "utils_gpu.cuh"
#include <thrust/device_vector.h>
#include <cusparse.h>
#include "common.cuh"
//! Rescale given column-stochastic graph, using specified lambda parameter
/*!
*/
void lambdaRescalingGPU( sparse_matrix<matval> P,        //!< Column-stocastic CSC matrix
                      matval lambda,          //!< λ rescaling parameter
                      bool dist=false,        //!< [optional] Consider input as distance?
                      bool dropLeafEdge=false //!< [optional] Remove edges from leaf nodes?
                      );
uint32_t makeStochasticGPU(sparse_matrix<matval> *P);
sparse_matrix<matval>* symmetrizeMatrixGPU(sparse_matrix<matval> *A, cusparseHandle_t &handle);
uint32_t makeStochasticGPU(matval* val,int* row, int n);
int SymmetrizeMatrix(cusparseHandle_t &handle,
        float** symmetrized_values,
        int** symmetrized_rowptr,
        int** symmetrized_colind,
        float *d_values,
        int* d_indices,
        const int num_points,
       int *row_ptr,const int nnz);
#endif
