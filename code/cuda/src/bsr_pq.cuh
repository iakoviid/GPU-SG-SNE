#ifndef BSR_PQ_CUH
#define BSR_PQ_CUH
#include <cuda_runtime.h>
template <typename data_type, typename index_type>
__global__ void bsrpq_r(index_type n_block_rows, index_type bs,
                     const index_type *__restrict__ col_ids,
                     const index_type *__restrict__ row_ptr,
                     const data_type *__restrict__ data,
                     const data_type *__restrict__ Y, data_type *Fattr,
                     index_type n, index_type d) {
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = idx % bs;
  const index_type block_row = idx / bs;
  data_type dist;
  if (block_row >= n_block_rows) {return;}
    const index_type first_block = row_ptr[block_row];
    const index_type last_block = row_ptr[block_row + 1];

    if (row < bs && block_row < n_block_rows) {
      data_type sum1 = 0;
      data_type sum2 = 0;
      data_type sum3 = 0;

      for (index_type block = first_block; block < last_block; block++) {
        const index_type first_col = col_ids[block] * bs;
        for (index_type col = 0; col < bs; col++) {
          index_type row_v = block_row * bs + row;
          index_type column = first_col + col;
          dist = 0;
          for (int dim = 0; dim < d; dim++) {
            dist += (Y[row_v + dim * n] - Y[column + dim * n]) *
                    (Y[row_v + dim * n] - Y[column + dim * n]);
          }
          data_type pq = data[block * bs * bs + row * bs + col] / (1 + dist);
          switch (d) {
          case 1:
            sum1 += pq * (Y[row_v] - Y[column]);
            break;
          case 2:
            sum1 += pq * (Y[row_v] - Y[column]);
            sum2 += pq * (Y[row_v + n] - Y[column + n]);
            break;
          case 3:
            sum1 += pq * (Y[row_v] - Y[column]);
            sum2 += pq * (Y[row_v + n] - Y[column + n]);
            sum3 += pq * (Y[row_v + 2 * n] - Y[column + 2 * n]);
            break;
          }
        }
      }
      switch (d) {
      case 1:
        Fattr[block_row * bs + row] = sum1;
        break;
      case 2:
        Fattr[block_row * bs + row] = sum1;
        Fattr[block_row * bs + row + n] = sum2;
        break;
      case 3:
        Fattr[block_row * bs + row] = sum1;
        Fattr[block_row * bs + row + n] = sum2;
        Fattr[block_row * bs + row + 2 * n] = sum3;
        break;
      }
    }

}

template <typename index_type>
__device__ index_type round_up_to_power_of_two(index_type v) {
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}
template <class T> struct shared_memory {
  __device__ inline operator T *() {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};
template <typename data_type, typename index_type, index_type bs>
__global__ void
bsr_col(index_type n_block_rows,const index_type *__restrict__ col_ids,
     const index_type *__restrict__ row_ptr, const data_type *__restrict__ data,
     const data_type *__restrict__ Y, data_type *__restrict__ Fattr,
     index_type n, index_type d) {

  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
  if (block_row >= n_block_rows) {return;}
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];
  data_type dist;

  index_type col = first_block * bs + lane / bs;
  index_type r = lane % bs;

  data_type *partial_sums =
      shared_memory<data_type>(); ///< Size is equal to blockDim.x *
                                  ///< sizeof(data_type)

  data_type sum1 = 0;
  data_type sum2 = 0;
  data_type sum3 = 0;

  for (; col < last_block * bs; col += 32 / bs) {
    const index_type block = col / bs;
    const index_type c = col % bs;
    int column = col_ids[block] * bs + c;
    int row = block_row * bs + r;
    dist = 0;
    for (int dim = 0; dim < d; dim++) {
      dist += (Y[row + dim * n] - Y[column + dim * n]) *
              (Y[row + dim * n] - Y[column + dim * n]);
    }
    const data_type value = data[block * bs * bs + c * bs + r] /
                            (1 + dist); //<<---------------------------------
    switch (d) {
    case 1:
      sum1 += value * (Y[row] - Y[column]);
      break;
    case 2:
      sum1 += value * (Y[row] - Y[column]);
      sum2 += value * (Y[row + n] - Y[column + n]);
      break;
    case 3:
      sum1 += value * (Y[row] - Y[column]);
      sum2 += value * (Y[row + n] - Y[column + n]);
      sum3 += value * (Y[row + 2 * n] - Y[column + 2 * n]);
      break;
    }
  }
  switch (d) {
  case 1:
    partial_sums[threadIdx.x] = sum1;
    break;
  case 2:
    partial_sums[threadIdx.x] = sum1;
    partial_sums[threadIdx.x + blockDim.x] = sum2;
    break;
  case 3:
    partial_sums[threadIdx.x] = sum1;
    partial_sums[threadIdx.x + blockDim.x] = sum2;
    partial_sums[threadIdx.x + 2 * blockDim.x] = sum3;
    break;
  }

  for (index_type stride = round_up_to_power_of_two((32 / bs) / 2); stride > 0;
       stride /= 2) {

    __syncthreads();
    if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
      for (int dim = 0; dim < d; dim++) {
        partial_sums[threadIdx.x + blockDim.x * dim] +=
            partial_sums[threadIdx.x + stride * bs + blockDim.x * dim];
      }
  }

  if (lane < bs)
    for (int dim = 0; dim < d; dim++) {
      Fattr[block_row * bs + lane + dim * n] =
          partial_sums[threadIdx.x + blockDim.x * dim];
    }
}
#endif
