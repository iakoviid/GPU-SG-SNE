#ifndef PQ_CUH
#define PQ_CUH
#include "common.cuh"
#include "hybrid.cuh"
#include <cuda_runtime.h>
template<class dataPoint>
dataPoint tsneCost(sparse_matrix<dataPoint>P,dataPoint* y, int n,int d,dataPoint alpha,dataPoint zeta );
template <class dataPoint>
void AttractiveEstimation(sparse_matrix<dataPoint> P, int d, dataPoint *y,
                          dataPoint *Fattr);
template <class dataPoint>
void AttractiveEstimation(int *row, int *col, dataPoint *data, dataPoint *Fattr,
                          dataPoint *y, int n, int d, int blockSize,
                          int blockRows, int num_blocks, int nnz, int format);
// Parallel SpMV using CSR format
template <typename data_type, typename index_type>
__global__ void pq_csr_naive(const index_type *row_ptr,
                             const index_type *col_ind, const data_type *values,
                             const index_type num_rows, const index_type d,
                             const data_type *Y, data_type *Fattr) {
  // Uses a grid-stride loop to perform dot product
  for (index_type i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rows;
       i += blockDim.x * gridDim.x) {
    data_type sum1 = 0;
    data_type sum2 = 0;
    data_type sum3 = 0;
    data_type dist = 0;

    const index_type row_start = row_ptr[i];
    const index_type row_end = row_ptr[i + 1];

    for (index_type j = row_start; j < row_end; j++) {
      index_type row = i;
      index_type column = col_ind[j];
      dist = 0;
      for (int dim = 0; dim < d; dim++) {
        dist += (Y[row + dim * num_rows] - Y[column + dim * num_rows]) *
                (Y[row + dim * num_rows] - Y[column + dim * num_rows]);
      }
      data_type pq = values[j] / (1 + dist);

      switch (d) {
      case 1:
        sum1 += pq * (Y[row] - Y[column]);
        break;
      case 2:
        sum1 += pq * (Y[row] - Y[column]);
        sum2 += pq * (Y[row + num_rows] - Y[column + num_rows]);

        break;
      case 3:
        sum1 += pq * (Y[row] - Y[column]);
        sum2 += pq * (Y[row + num_rows] - Y[column + num_rows]);
        sum3 += pq * (Y[row + 2 * num_rows] - Y[column + 2 * num_rows]);
        break;
      }
    }
    switch (d) {
    case 1:
      Fattr[i] = sum1;
      break;
    case 2:
      Fattr[i] = sum1;
      Fattr[i + num_rows] = sum2;

      break;
    case 3:
      Fattr[i] = sum1;
      Fattr[i + num_rows] = sum2;
      Fattr[i + 2 * num_rows] = sum3;

      break;
    }
  }
}

template <typename data_type,int d>
__global__ void
coo_pq_kernel(const int n_elements, const matidx *__restrict__ col_ids,
              const matidx *__restrict__ row_ids,
              const data_type *__restrict__ data,
              const data_type *__restrict__ Y, data_type *__restrict__ Fattr, const int n) {
  register matidx row, column;
  register data_type dist, pq;
  register unsigned int element;
  register data_type dx, dy, dz;
  for (element = blockIdx.x * blockDim.x + threadIdx.x; element < n_elements;
       element += blockDim.x * gridDim.x) {
    row = row_ids[element];
    column = col_ids[element];
    switch (d) {
    case 1:
      dx = (Y[row] - Y[column]);
      dist = dx * dx;
      pq = data[element] / (1 + dist);
      atomicAdd(Fattr + row, pq * dx);
      break;
    case 2:
      dx = (Y[row] - Y[column]);
      dy = (Y[row + n] - Y[column + n]);
      dist = dx * dx + dy * dy;
      pq = data[element] / (1 + dist);
      atomicAdd(Fattr + row, pq * dx);
      atomicAdd(Fattr + row + n, pq * dy);
      break;
    case 3:
      dx = (Y[row] - Y[column]);
      dy = (Y[row + n] - Y[column + n]);
      dz = (Y[row + 2 * n] - Y[column + 2 * n]);
      dist = dx * dx + dy * dy + dz * dz;
      pq = data[element] / (1 + dist);
      atomicAdd(Fattr + row, pq * dx);
      atomicAdd(Fattr + row + n, pq * dy);
      atomicAdd(Fattr + row + 2 * n, pq * dz);
      break;
    }
  }
}
#endif
