#ifndef PQ_CUH
#define PQ_CUH
#include <cuda_runtime.h>
#include "common.cuh"
#include "bsr_pq.cuh"

void AttractiveEstimation(int *row, int *col, coord *data,
                          coord  *Fattr, coord *y, int n,
                          int d, int blockSize,int blockRows,int num_blocks,int nnz,int format);
// Parallel SpMV using CSR format
template <typename data_type, typename index_type>
__global__ void pq_csr_naive(const index_type *row_ptr, const index_type *col_ind, const data_type *values, const index_type num_rows,const index_type d, const data_type *Y, data_type *Fattr) {
    // Uses a grid-stride loop to perform dot product
    for (index_type i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows; i += blockDim.x * gridDim.x) {
        data_type sum1 = 0;
        data_type sum2 = 0;
        data_type sum3 = 0;
        data_type dist = 0;

        const index_type row_start = row_ptr[i];
        const index_type row_end = row_ptr[i + 1];

        for (index_type j = row_start; j < row_end; j++) {
          index_type row =i;
          index_type column =col_ind[j];
          dist = 0;
          for (int dim = 0; dim < d; dim++) {
            dist += (Y[row + dim * num_rows] - Y[column + dim * num_rows]) *
                    (Y[row + dim * num_rows] - Y[column + dim * num_rows]);
          }
          data_type pq = values[j] / (1 + dist);

          switch (d) {
          case 1:
            sum1+=pq*(Y[row] - Y[column]);
            break;
          case 2:
            sum1+=pq*(Y[row] - Y[column]);
            sum2+=pq*(Y[row+num_rows] - Y[column+num_rows]);

            break;
          case 3:
            sum1+=pq*(Y[row] - Y[column]);
            sum2+=pq*(Y[row+num_rows] - Y[column+num_rows]);
            sum3+=pq*(Y[row+2*num_rows] - Y[column+2*num_rows]);
            break;
          }
        }
        switch (d) {
        case 1:
          Fattr[i]=sum1;
          break;
        case 2:
          Fattr[i]=sum1;
          Fattr[i+num_rows]=sum2;

          break;
        case 3:
          Fattr[i]=sum1;
          Fattr[i+num_rows]=sum2;
          Fattr[i+2*num_rows]=sum3;

          break;
        }

    }
 }

 template <typename data_type>
 __global__ void coo_pq_kernel (
      int n_elements,
     const matidx *col_ids,
     const matidx *row_ids,
     const data_type*data,
     const data_type*Y,
     data_type*Fattr,int d,int n)
 {
   unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;

   if (element < n_elements)
   {
     data_type dist = 0;
     matidx row=row_ids[element];
     matidx column =col_ids[element];
     for (int dim = 0; dim < d; dim++) {
       dist += (Y[row + dim * n] - Y[column + dim * n]) *
               (Y[row + dim * n] - Y[column + dim * n]);
     }
     data_type pq = data[element] / (1 + dist);
     switch (d) {
     case 1:
       atomicAdd(Fattr+ row,pq*(Y[row] - Y[column]));
       break;
     case 2:
       atomicAdd(Fattr+ row,pq*(Y[row] - Y[column]));
       atomicAdd(Fattr+ row+n,pq*(Y[row+n] - Y[column+n]));

       break;
     case 3:
       atomicAdd(Fattr+ row,pq*(Y[row] - Y[column]));
       atomicAdd(Fattr+ row+n,pq*(Y[row+n] - Y[column+n]));
       atomicAdd(Fattr+ row+2*n,pq*(Y[row+2*n] - Y[column+2*n]));

       break;
     }
   }
 }
 #endif
