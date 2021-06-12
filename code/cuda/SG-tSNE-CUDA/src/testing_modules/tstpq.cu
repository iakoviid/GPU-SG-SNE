#include <cuda_runtime.h>
#include <cusparse.h>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <sys/time.h>
#include "../prepareMatrix.cuh"
#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define ANSI_COLOR_RED "\x1b[31m"
using namespace std;

float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
  return dis(e);
}

int get_random_int(int max) {
  static std::default_random_engine e;
  std::uniform_int_distribution<int> dis(0, max - 1);
  return dis(e);
}

void generate_candidate_blocks(int R, int C, int BS_R, int BS_C, int num_blocks,
                               int *weight_indptr, int *weight_indices) {
  std::map<int, std::set<int>> blocks;
  int num_r_block = R;
  int num_c_block = C;
  int curr_size = 0;
  while (curr_size < num_blocks) {
    int r = get_random_int(num_r_block);
    int c = get_random_int(num_c_block);
    if (blocks[r].count(c) == 0) {
      blocks[r].insert(c);
      curr_size++;
    }
  }

  int current_ptr = 0;
  int i;
  for (i = 0; i < num_r_block; i++) {
    weight_indptr[i] = current_ptr;
    for (auto block : blocks[i]) {
      weight_indices[current_ptr++] = block;
    }
  }
  weight_indptr[i] = current_ptr;
}

void serial(float *val, int *block_ptr, int *col_ind, float *Y, float *Fattr,
            int n, int bs, int d,int m) {
  for (int i = 0; i < n; i++) {
    int block_first = block_ptr[i];
    int block_last = block_ptr[i + 1];
    for (int block = block_first; block < block_last; block++) {
      for (int row = 0; row < bs; row++) {
        for (int col = 0; col < bs; col++) {
          // printf("%d  %d  %f \n", i * bs + row,
          int row_v = i * bs + row;

          int column = col_ind[block] * bs + col;
          float dist = 0;
          for (int dim = 0; dim < d; dim++) {
            dist += (Y[row_v + dim * m] - Y[column + dim * m]) *
                    (Y[row_v + dim * m] - Y[column + dim * m]);
          }
          for (int dim = 0; dim < d; dim++) {
            Fattr[row_v + m * dim] +=
                val[block * bs * bs + row * bs + col] *
                (Y[row_v + m * dim] - Y[column + m * dim]) / (dist + 1);
          }
          //if (row_v==0){
            //printf("%f %f\n",Fattr[0], Fattr[m] );
          //}

        }
      }
    }
  }

}

void spmv_cusparse(float *val, int *block_ptr, int *col_ind, float *x, float *y,
                   int n, int blockSize, int num_blocks, int rows) {
  // For blas 2 gemv y = alpha.x.A + Beta.y
  const float alpha = 1.0;
  const float beta = 0.0;
  float *val_d;
  int *col_ind_d, *block_ptr_d;
  cudaMalloc(&val_d, num_blocks * blockSize * blockSize * sizeof(float));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(int));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(int));
  cudaMemcpy(val_d, val, num_blocks * blockSize * blockSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);

  // Copy input
  float *cu_x;
  cudaMalloc(&cu_x, rows * sizeof(float));
  cudaMemcpy(cu_x, x, rows * sizeof(float), cudaMemcpyHostToDevice);

  float *cu_y;
  cudaMalloc(&cu_y, rows * sizeof(float));

  // Init matrix properties
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv

  cusparseSbsrmv(handle, CUSPARSE_DIRECTION_COLUMN,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, num_blocks, &alpha,
                 descr, val, block_ptr, col_ind, blockSize, cu_x, &beta, cu_y);

  // Get back result
  cudaMemcpy(y, cu_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
  // Dealloc vectors
  // CUDA_CALL(cudaFree(cu_x));
  // CUDA_CALL(cudaFree(cu_y));
}

template <typename data_type, typename index_type>
__global__ void bsr1(index_type n_block_rows, index_type bs,
                     const index_type *__restrict__ col_ids,
                     const index_type *__restrict__ row_ptr,
                     const data_type *__restrict__ data,
                     const data_type *__restrict__ Y, data_type *Fattr,
                     index_type n, index_type d) {
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = idx % bs;
  const index_type block_row = idx / bs;
  data_type dist;
  if (block_row < n_block_rows) {
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
}

void bsr1run(float *val, int *block_ptr, int *col_ind, float *y, float *Fattr,
             int n, int blockSize, int num_blocks, int rows, int d) {
  float *cu_y;
  float *cu_F;
  CUDA_CALL(cudaMalloc(&cu_y, rows * d * sizeof(float)));
  CUDA_CALL(
      cudaMemcpy(cu_y, y, rows * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cu_F, rows * d * sizeof(float)));
  CUDA_CALL(cudaMemcpy(cu_F, Fattr, rows * d * sizeof(float),
                       cudaMemcpyHostToDevice));

  dim3 block_size = 512;
  dim3 grid_size{};
  grid_size.x = (n * blockSize + block_size.x - 1) / block_size.x;
  float *val_d;
  int *col_ind_d, *block_ptr_d;
  cudaMalloc(&val_d, num_blocks * blockSize * blockSize * sizeof(float));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(int));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(int));
  cudaMemcpy(val_d, val, num_blocks * blockSize * blockSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);
  // bcsr_spmv_kernel_column_by_column_template<<<grid_size, block_size,
  // block_size.x *
  // sizeof(double)>>>(bcsr.cu_bsrColIndC,bcsr.cu_bsrRowPtrC,bcsr.cu_bsrValC,cu_x,cu_y);
  bsr1<float, int><<<grid_size, block_size>>>(
      n, blockSize, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);

  cudaMemcpy(Fattr, cu_F, rows * d * sizeof(float), cudaMemcpyDeviceToHost);
  // CUDA_CALL(cudaFree(cu_x));
  // CUDA_CALL(cudaFree(cu_y));
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
bsr2(const index_type *__restrict__ col_ids,
     const index_type *__restrict__ row_ptr, const data_type *__restrict__ data,
     const data_type *__restrict__ Y, data_type *__restrict__ Fattr,
     index_type n, index_type d) {

  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
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

template <typename data_type, typename index_type>
void transpose_blocks (data_type *new_values,index_type* row_ptr,index_type bs,index_type n_rows,data_type* values)
{
  std::unique_ptr<data_type[]> buffer (new data_type[bs * bs]);

  for (index_type row = 0; row < n_rows; row++)
    {
      for (index_type block = row_ptr[row]; block < row_ptr[row + 1]; block++)
        {
          data_type *new_block_data = new_values + bs * bs * block;
          data_type *old_block_data = values+ bs * bs * block;
          std::copy_n (old_block_data, bs * bs, buffer.get ());

          for (unsigned int i = 0; i < bs; i++)
            for (unsigned int j = 0; j < bs; j++)
              new_block_data[j * bs + i] = buffer[i * bs + j];
        }
    }
}
void bsr2run(float *val, int *block_ptr, int *col_ind, float *y, float *Fattr,
             int n, int blockSize, int num_blocks, int rows, int d) {

  float *cu_y;
  float *cu_F;
  CUDA_CALL(cudaMalloc(&cu_y, rows * d * sizeof(float)));
  CUDA_CALL(
      cudaMemcpy(cu_y, y, rows * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cu_F, rows * d * sizeof(float)));
  CUDA_CALL(cudaMemcpy(cu_F, Fattr, rows * d * sizeof(float),
                       cudaMemcpyHostToDevice));
   float* val_t=(float *)malloc(sizeof(float)*num_blocks*blockSize*blockSize);
   transpose_blocks<float,int>(val_t, block_ptr,blockSize,n,val);

  float *val_d;
  int *col_ind_d, *block_ptr_d;
  cudaMalloc(&val_d, num_blocks * blockSize * blockSize * sizeof(float));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(int));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(int));
  cudaMemcpy(val_d, val_t, num_blocks * blockSize * blockSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 block_size = 32;
  dim3 grid_size{};

  grid_size.x = (n * 32 + block_size.x - 1) / block_size.x;
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  switch (blockSize) {
  case 1:
    bsr2<float, int, 1>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 2:
    bsr2<float, int, 2>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 3:
    bsr2<float, int, 3>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 4:
    bsr2<float, int, 4>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 8:
    bsr2<float, int, 8>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 16:
    bsr2<float, int, 16>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 32:
    bsr2<float, int, 32>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("BSR2 elapsedTime=%lf\n",elapsedTime );


  cudaMemcpy(Fattr, cu_F, rows * d * sizeof(float), cudaMemcpyDeviceToHost);
}

template <typename data_type, typename index_type, index_type bs>
__global__ void
bsr2New(const index_type *__restrict__ col_ids,
     const index_type *__restrict__ row_ptr, const data_type *__restrict__ data,
     const data_type *__restrict__ Y, data_type *__restrict__ Fattr,
     index_type n, index_type d) {

  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
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

void bsr2runNew(float *val, int *block_ptr, int *col_ind, float *y, float *Fattr,
             int n, int blockSize, int num_blocks, int rows, int d) {

  float *cu_y;
  float *cu_F;
  CUDA_CALL(cudaMalloc(&cu_y, rows * d * sizeof(float)));
  CUDA_CALL(
      cudaMemcpy(cu_y, y, rows * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cu_F, rows * d * sizeof(float)));
  CUDA_CALL(cudaMemcpy(cu_F, Fattr, rows * d * sizeof(float),
                       cudaMemcpyHostToDevice));
   float* val_t=(float *)malloc(sizeof(float)*num_blocks*blockSize*blockSize);
   transpose_blocks<float,int>(val_t, block_ptr,blockSize,n,val);

  float *val_d;
  int *col_ind_d, *block_ptr_d;
  cudaMalloc(&val_d, num_blocks * blockSize * blockSize * sizeof(float));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(int));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(int));
  cudaMemcpy(val_d, val_t, num_blocks * blockSize * blockSize * sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 block_size = 32;
  dim3 grid_size{};

  grid_size.x = (n * 32 + block_size.x - 1) / block_size.x;
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  switch (blockSize) {
  case 1:
    bsr2New<float, int, 1>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 2:
    bsr2New<float, int, 2>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 3:
    bsr2New<float, int, 3>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 4:
    bsr2New<float, int, 4>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 8:
    bsr2New<float, int, 8>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 16:
    bsr2New<float, int, 16>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 32:
    bsr2New<float, int, 32>
        <<<grid_size, block_size, block_size.x * d * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("BSR2 elapsedTime=%lf\n",elapsedTime );


  cudaMemcpy(Fattr, cu_F, rows * d * sizeof(float), cudaMemcpyDeviceToHost);
}
template <class dataPoint>
dataPoint maxerror(dataPoint *const w, dataPoint *v, int n, int d) {

  dataPoint maxError = 0;
  dataPoint avgError = 0;
  int pos = 0;

  for (int i = 0; i < n ; i++) {
    for(int j=0;j<d;j++){
    if ((v[i+j*n] - w[i*d+j]) * (v[i+j*n] - w[i*d+j]) > maxError) {
      maxError = (v[i+j*n] - w[i*d+j]) * (v[i+j*n] - w[i*d+j]);
      pos = i;
    }
    avgError += (v[i+j*n] - w[i*d+j]) * (v[i+j*n] - w[i*d+j]);
  }}

  printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d size=%d\n",
         maxError, pos, v[pos], w[pos], avgError / (n * d), n, n * d);

  return maxError;
}
void test_pq(float *val, int *block_ptr, int *col_ind, float *y, int n,
             int blockSize, int num_blocks, int rows, int d) {
  // float *ysparse=(float* )malloc(sizeof(float)*rows);
  // spmv_cusparse(val,block_ptr,col_ind,x,ysparse,n,blockSize,num_blocks,rows);
  // for(int i=0;i<rows;i++){
  //  printf("%.1f ",ysparse[i]);
  //}
  // printf("\n" );
  float *Fserial = (float *)malloc(sizeof(float) * rows * d);
/*
  for (int i = 0; i < rows; i++) {
    printf("%5d", i);
  }
  printf("\n");
*/
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      Fserial[i + j * rows] = 0;
    }
  }
  serial(val, block_ptr, col_ind, y, Fserial, n, blockSize, d, rows);
/*
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < rows; i++) {
      printf("%5.1f", Fserial[i + j * rows]);
    }
    printf("\n");
  }
  printf("\n");
*/
  float *Fbsr1 = (float *)malloc(sizeof(float) * rows * d);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      Fbsr1[i + j * rows] = 0;
    }
  }
  bsr1run(val, block_ptr, col_ind, y, Fbsr1, n, blockSize, num_blocks, rows, d);
  maxerror(Fserial,Fbsr1,rows,d);
/*
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < rows; i++) {
      printf("%5.1f", Fbsr1[i + j * rows]);
    }
    printf("\n");
  }
  printf("\n");
*/
  float *Fbsr2 = (float *)malloc(sizeof(float) * rows * d);
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      Fbsr2[i + j * rows] = 0;
    }
  }

  bsr2run(val, block_ptr, col_ind, y, Fbsr2, n, blockSize, num_blocks, rows, d);
  maxerror(Fserial,Fbsr2,rows,d);

/*
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < rows; i++) {
      printf("%5.1f", Fbsr2[i + j * rows]);
    }
    printf("\n");
  }

  printf("\n");
*/

}

void pqF(float *Fattr, float *const Y, float const *const p_sp, int *ir, int *jc,
        int const n, int const d) {
  for (int i = 0; i < n * d; i++) {
    Fattr[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    // for (unsigned int j = 0; j < n; j++) {

    double accum[3] = {0};
    double Yj[3];
    double Yi[3];
    double Ftemp[3]={0};

    const int k =
        ir[i + 1] - ir[i]; /* number of nonzero elements of each row */
    for (int x = 0; x < d; x++) {
      Yi[x] = Y[i  + x*n];
    }
    /* for each non zero element */
    for (unsigned int idx = 0; idx < k; idx++) {

      const unsigned int j = (jc[ir[i] + idx]);

      for (int x = 0; x < d; x++) {
        Yj[x] = Y[j  + x*n];
      }
      /* distance computation */
      double dist = 0;
      for (int x = 0; x < d; x++) {
        dist += (Yi[x] - Yj[x]) * (Yi[x] - Yj[x]);
      }
      /* P_{ij} \times Q_{ij} */
      const double p_times_q = p_sp[ir[i] + idx] / (1 + dist);
      for (int x = 0; x < d; x++) {
        Ftemp[x] += p_times_q * (Yi[x] - Yj[x]);
      }
    }
    for (int x = 0; x < d; x++) {

      Fattr[((i)) + x*n] = Ftemp[x];
      //printf("Fattr[%d+%d*%d]= %f\n",i,x,n,Fattr[i+x*n] );
    }
  }
}

void test_csrpq(float *val, int* col, int* rptr, int n, int nnz, float* Y, int d) {


  float *Fserial = (float *)malloc(sizeof(float) * n * d);
/*
  for (int i = 0; i < n; i++) {
    printf("%5d", i);
  }
  printf("\n");
*/
  pqF(Fserial, Y, val, rptr, col, n, d);
/*
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < n; i++) {
      printf("%5.1f", Fserial[i + j * n]);
    }
    printf("\n");
  }
*/
  //float *Fwarp=(float *)malloc(sizeof(float)*n*d);
}
void bsr2csr(float* bsrValA,int* bsrRowPtrA,int* bsrColIndA, int n,int bs,int num_blocks, int rows,float* csrVal,int* csrRptr,int* csrCidx){
  // Given BSR format (bsrRowPtrA, bsrcolIndA, bsrValA) and
  // blocks of BSR format are stored in column-major order.
  cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
  int m = n*bs;
  //int nnzb = bsrRowPtrA[n] - bsrRowPtrA[0]; // number of blocks
  int nnz  = num_blocks * bs * bs; // number of elements
  int* csrRowPtrC,* csrColIndC;
  float* csrValC;
  cudaMalloc((void**)&csrRowPtrC, sizeof(int)*(m+1));
  cudaMalloc((void**)&csrColIndC, sizeof(int)*nnz);
  cudaMalloc((void**)&csrValC, sizeof(float)*nnz);
  float *bsrValA_d;
  int *bsrColIndA_d, *bsrRowPtrA_d;
  cudaMalloc(&bsrValA_d, nnz* sizeof(float));
  cudaMalloc(&bsrRowPtrA_d, (n + 1) * sizeof(int));
  cudaMalloc(&bsrColIndA_d, num_blocks * sizeof(int));
  cudaMemcpy(bsrValA_d, bsrValA, nnz* sizeof(float),
             cudaMemcpyHostToDevice);
  cudaMemcpy(bsrRowPtrA_d, bsrRowPtrA, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(bsrColIndA_d, bsrColIndA, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseSbsr2csr(handle, dir, n, n,
          descr,
          bsrValA_d, bsrRowPtrA_d, bsrColIndA_d,
          bs,
          descr,
          csrValC, csrRowPtrC, csrColIndC);
  cudaMemcpy(csrVal,csrValC,nnz*sizeof(float),cudaMemcpyDeviceToHost);
  cudaMemcpy(csrCidx,csrColIndC,nnz*sizeof(int),cudaMemcpyDeviceToHost);
  cudaMemcpy(csrRptr,csrRowPtrC,(rows+1)*sizeof(int),cudaMemcpyDeviceToHost);



}
__global__ void ComputePijxQijKernel(float * __restrict__ attr_forces,const float * __restrict__ pij,const float * __restrict__ points,
                            const int * __restrict__ coo_indicesi,const int * __restrict__ coo_indicesj,
                            const int num_points,
                            const int num_nonzero)
{
    register int TID, i, j;
    register float ix, iy, jx, jy, dx, dy, pijqij;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_nonzero) return;
    i = coo_indicesi[TID];
    j = coo_indicesj[TID];

    ix = points[i]; iy = points[num_points + i];
    jx = points[j]; jy = points[num_points + j];
    dx = ix - jx;
    dy = iy - jy;
    pijqij = pij[TID] / (1 + dx*dx + dy*dy);
    atomicAdd(attr_forces + i, pijqij * dx);
    atomicAdd(attr_forces + num_points + i, pijqij * dy);
}
__host__ __device__ int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }
void tsne_cuda_spmv(float* Fattr,float* coov,float* Y,int* cooi,int* cooj, int n,int nnz,int d){
  float* Fattr_d,*Y_d,*coov_d;
  int* cooi_d,*cooj_d;
  CUDA_CALL(cudaMalloc(&Fattr_d, n * d * sizeof(float)));
  CUDA_CALL(cudaMemcpy(Fattr_d, Fattr, n * d * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&Y_d, n * d * sizeof(float)));
  CUDA_CALL(cudaMemcpy(Y_d, Y, n * d * sizeof(float),cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc(&cooi_d, nnz * sizeof(int)));
  CUDA_CALL(cudaMemcpy(cooi_d, cooi, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cooj_d, nnz * sizeof(int)));
  CUDA_CALL(cudaMemcpy(cooj_d, cooj, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&coov_d, nnz * sizeof(float)));
  CUDA_CALL(cudaMemcpy(coov_d, coov, nnz * sizeof(float), cudaMemcpyHostToDevice));

  const int BLOCKSIZE = 1024;
  const int NBLOCKS = iDivUp(nnz, BLOCKSIZE);
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);

  ComputePijxQijKernel<<<NBLOCKS, BLOCKSIZE>>>(Fattr_d,coov_d, Y_d,cooi_d,cooj_d,n,nnz);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("COO elapsedTime=%lf\n",elapsedTime );
  CUDA_CALL(cudaMemcpy(Fattr, Fattr_d, n *d * sizeof(float), cudaMemcpyDeviceToHost));

}

void test_coo(float* coov,int* cooi,int* cooj,int nnz,float* Y, int n, int d){
  float *Frivals = (float *)malloc(sizeof(float) * n * d);
/*
  for (int i = 0; i < n; i++) {
    printf("%5d", i);
  }

  printf("\n" );
*/
  tsne_cuda_spmv(Frivals,coov, Y,cooi,cooj,n,nnz,d);
/*
  for (int j = 0; j < d; j++) {
    for (int i = 0; i < n; i++) {
      printf("%5.1f", Frivals[i + j * n]);
    }
    printf("\n");
}

printf("\n" );
*/
}


int main(int argc, char **argv) {

  ofstream myfile;
  ofstream myfile2;
  myfile.open("matrix.txt");
  myfile2.open("Y.txt");
  int N = 1 << atoi(argv[1]);
  int K = N;
  int bs = 1 << atoi(argv[2]);
  myfile << "N= " << N << " bs= " << bs << "\n";
  float density = (float)1 / (1 << (atoi(argv[3])));
  int d = atoi(argv[4]);
  printf("density=%f\n", density);
  float *weight;
  int *weight_ind;
  int *weight_ptr;
  int nnz = int(density * K * N * bs * bs);
  printf("nnz=%d\n", nnz);
  int num_blocks = int(nnz / (bs * bs)) + 1;
  printf("num_blocks=%d\n", num_blocks);
  weight = (float *)malloc(num_blocks * bs * bs * sizeof(float));
  weight_ind = (int *)malloc(num_blocks * sizeof(int));
  weight_ptr = (int *)malloc((N + 1) * sizeof(int));

  for (int i = 0; i < num_blocks * bs * bs; i++) {
    weight[i] = 10 * get_random();
  }

  generate_candidate_blocks(N, K, bs, bs, num_blocks, weight_ptr, weight_ind);
  int rows = N * bs;

  float *x = (float *)malloc(rows * d * sizeof(float));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      x[i + j * rows] = 10 * get_random();
      myfile2 << x[i + j * rows] << " ";
    }
    myfile2 << "\n";
  }
  test_pq(weight, weight_ptr, weight_ind, x, N, bs, num_blocks, rows, d);
  /*
    float** bsr=new float*[rows];
    for(int i = 0; i < rows; ++i)
        bsr[i] = new float[rows];
    for(int i=0;i<rows;i++){
      for(int j=0;j<rows;j++){
        bsr[i][j]=0;
      }
    }
  */
  float* coov=(float* )malloc(sizeof(float)*num_blocks*bs*bs);
  int* cooi=(int* )malloc(sizeof(float)*num_blocks*bs*bs);
  int* cooj=(int  * )malloc(sizeof(float)*num_blocks*bs*bs);
  int nnzcntr = 0;
  for (int i = 0; i < N; i++) {
    int block_first = weight_ptr[i];
    int block_last = weight_ptr[i + 1];
    for (int block = block_first; block < block_last; block++) {
      for (int row = 0; row < bs; row++) {
        for (int col = 0; col < bs; col++) {
          myfile << i * bs + row << " " << weight_ind[block] * bs + col << " "
                 << weight[block * bs * bs + row * bs + col] << "\n";
                 coov[nnzcntr]=weight[block * bs * bs + row * bs + col];
                 cooi[nnzcntr]=i * bs + row;
                 cooj[nnzcntr]=weight_ind[block] * bs + col ;

          nnzcntr++;
          // bsr[ i * bs + row][weight_ind[block] * bs + col]=weight[block * bs
          // * bs + row * bs + col];
        }
      }
    }
  }
  printf("nnzcntr=%d\n", nnzcntr);
  myfile.close();
  myfile2.close();
  float *csrVal = (float *)malloc(sizeof(float) * num_blocks * bs * bs);
  int *csrCidx = (int *)malloc(sizeof(int) * num_blocks * bs * bs);
  int *csrRptr = (int *)malloc(sizeof(int) * (rows+1));

  bsr2csr(weight, weight_ptr, weight_ind, N, bs, num_blocks, rows, csrVal,
          csrRptr, csrCidx);
   nnz = num_blocks * bs * bs;
   test_coo(coov,cooi,cooj,nnz,x,rows,d);

  test_csrpq(csrVal, csrCidx, csrRptr, rows, nnz, x, d);
  /*
  for(int i=0;i<rows;i++){
    for(int j=0;j<rows;j++){
      if(bsr[i][j]>0){
          printf("x " );
        }
          else{
        printf("o " );}


    }
    printf("\n" );
  }
*/

  return 0;
}
