#include <cuda_runtime.h>
#include <cusparse.h>

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <memory>
#include "./include/cuda_jit.h"
#define ANSI_COLOR_RED "\x1b[31m"
using namespace std;
static void CudaSparseCheckCore(cusparseStatus_t code, const char *file,
                                int line) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "Cuda Error %d : %s %s %d\n", code,
            cusparseGetErrorString(code), file, line);
    exit(code);
  }
}

#define CudaSparseCheck(test)                                                  \
  { CudaSparseCheckCore((test), __FILE__, __LINE__); }

static void CudaCheckCore(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda Error %d : %s %s %d\n", code,
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define CudaCheck(test)                                                        \
  { CudaCheckCore((test), __FILE__, __LINE__); }
#define CudaCheckAfterCall()                                                   \
  { CudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }
float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
  return dis(e);
}

int get_random_int(int max) {
  static std::default_random_engine e;
  std::uniform_int_distribution<int> dis(0, max-1);
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

void serial(float *val, int *block_ptr, int *col_ind, float *x, float *y, int n,
            int bs) {
  for (int i = 0; i < n; i++) {
    int block_first = block_ptr[i];
    int block_last = block_ptr[i + 1];
    for (int block = block_first; block < block_last; block++) {
      for (int row = 0; row < bs; row++) {
        for (int col = 0; col < bs; col++) {
          // printf("%d  %d  %f \n", i * bs + row,
          int row_v = i * bs + row;
          int column = col_ind[block] * bs + col;
          y[row_v] += val[block * bs * bs + row * bs + col] * x[column];
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
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv

  cusparseSbsrmv(handle, CUSPARSE_DIRECTION_COLUMN,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, num_blocks, &alpha,
                 descr, val, block_ptr, col_ind, blockSize, cu_x, &beta, cu_y);

  // Get back result
  cudaMemcpy(y, cu_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
  // Dealloc vectors
  // CudaCheck(cudaFree(cu_x));
  // CudaCheck(cudaFree(cu_y));
}

template <typename data_type, typename index_type>
__global__ void bsr1(index_type n_block_rows, index_type bs,
                     const index_type *__restrict__ col_ids,
                     const index_type *__restrict__ row_ptr,
                     const data_type *__restrict__ data,
                     const data_type *__restrict__ x, data_type *y) {
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = idx % bs;
  const index_type block_row = idx / bs;
  if (block_row < n_block_rows) {
    const index_type first_block = row_ptr[block_row];
    const index_type last_block = row_ptr[block_row + 1];

    if (row < bs && block_row < n_block_rows) {
      data_type local_out = 0.0;

      for (index_type block = first_block; block < last_block; block++) {
        const index_type first_col = col_ids[block] * bs;
        for (index_type col = 0; col < bs; col++)
          local_out +=
              x[first_col + col] * data[block * bs * bs + row * bs + col];
      }

      y[block_row * bs + row] = local_out;
    }
  }
}

void bsr1run(float *val, int *block_ptr, int *col_ind, float *x, float *y,
             int n, int blockSize, int num_blocks, int rows) {
  float *cu_x;
  float *cu_y;
  CudaCheck(cudaMalloc(&cu_x, rows * sizeof(float)));
  CudaCheck(cudaMemcpy(cu_x, x, rows * sizeof(float), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&cu_y, rows * sizeof(float)));
  CudaCheck(cudaMemcpy(cu_y, y, rows * sizeof(float), cudaMemcpyHostToDevice));

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
  bsr1<float, int><<<grid_size, block_size>>>(n, blockSize, col_ind_d,
                                              block_ptr_d, val_d, cu_x, cu_y);

  cudaMemcpy(y, cu_y, rows * sizeof(float), cudaMemcpyDeviceToHost);
  // CudaCheck(cudaFree(cu_x));
  // CudaCheck(cudaFree(cu_y));
}
template <typename index_type>
__device__ index_type round_up_to_power_of_two (index_type v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}
template<class T>
struct shared_memory
{
  __device__ inline operator T *()
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }

  __device__ inline operator const T *() const
  {
    extern __shared__ int __smem[];
    return (T *)__smem;
  }
};
template <typename data_type, typename index_type, index_type bs>
__global__ void bsr2(
    const index_type *__restrict__ col_ids,
    const index_type *__restrict__ row_ptr, const data_type *__restrict__ data,
    const data_type *__restrict__ x, data_type *__restrict__ y) {
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  index_type col = first_block * bs + lane / bs;
  index_type r = lane % bs;

  data_type *partial_sums =
      shared_memory<data_type>(); ///< Size is equal to blockDim.x *
                                  ///< sizeof(data_type)

  data_type local_out = 0.0;

  for (; col < last_block * bs; col += 32 / bs) {
    const index_type block = col / bs;
    const index_type c = col % bs;

    const data_type value = data[block * bs * bs + r * bs + c];//<<---------------------------------
    const data_type x_value = x[col_ids[block] * bs + c];
    local_out += x_value * value;
  }

  partial_sums[threadIdx.x] = local_out;

  for (index_type stride = round_up_to_power_of_two((32 / bs) / 2); stride > 0;
       stride /= 2) {
    __syncthreads();
    if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
      partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
  }

  if (lane < bs)
    y[block_row * bs + lane] = partial_sums[threadIdx.x];
}

void bsr2run(float *val, int *block_ptr, int *col_ind, float *x, float *y,
             int n, int blockSize, int num_blocks, int rows) {
               float *cu_x;
               float *cu_y;
               CudaCheck(cudaMalloc(&cu_x, rows * sizeof(float)));
               CudaCheck(cudaMemcpy(cu_x, x, rows * sizeof(float), cudaMemcpyHostToDevice));
               CudaCheck(cudaMalloc(&cu_y, rows * sizeof(float)));
               CudaCheck(cudaMemcpy(cu_y, y, rows * sizeof(float), cudaMemcpyHostToDevice));

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

  dim3 block_size = 32;
  dim3 grid_size{};

  grid_size.x = (n * 32 + block_size.x - 1) / block_size.x;

  switch (blockSize) {
  case 1:
    bsr2<float, int, 1>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  case 2:
    bsr2<float, int, 2>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  case 3:
    bsr2<float, int, 3>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  case 4:
    bsr2<float, int, 4>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  case 8:
    bsr2<float, int, 8>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  case 16:
    bsr2<float, int, 16>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  case 32:
    bsr2<float, int, 32>
        <<<grid_size, block_size, block_size.x * sizeof(float)>>>(
            col_ind_d, block_ptr_d, val_d, cu_x, cu_y);
    break;
  }
  cudaMemcpy(y, cu_y, rows * sizeof(float), cudaMemcpyDeviceToHost);

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

template <typename data_type, typename index_type>
void spmv_jit(data_type *val, index_type *block_ptr, index_type *col_ind,
              data_type *x, data_type *y,index_type n, index_type bs, index_type num_blocks, index_type rows) {

  const index_type n_rows = n;
  dim3 block_size = 32;
  dim3 grid_size{};

  data_type* val_t=(data_type *)malloc(sizeof(data_type)*num_blocks*bs*bs);
  transpose_blocks<data_type,index_type>(val_t, block_ptr,bs,rows,val);

  grid_size.x = (n * 32 + block_size.x - 1) / block_size.x;

  jit(bcsr_jit,
  {
    const int bs = {{ bs }};

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx % 32;
    const int block_row = idx / 32; ///< Warp per block row
    const int first_block = row_ptr[block_row];
    const int last_block = row_ptr[block_row + 1];

    int col = first_block * bs + lane / bs;
    int r = lane % bs;

    __shared__ float partial_sums[{{ shared_size }}]; // = shared_memory<float> (); ///< Size is equal to blockDim.x * sizeof(float)

    float local_out = 0.0;

    for (; col < last_block * bs; col += 32 / bs)
      {
        const int block = col / bs;
        const int c = col % bs;

        const float value = data[block * bs * bs + c * bs + r];
        const float x_value = x[col_ids[block] * bs + c];
        local_out += x_value * value;
      }

    partial_sums[threadIdx.x] = local_out;

    for (int stride = {{ stride_begin }} ; stride > 0; stride /= 2)
      {
        __syncthreads ();
        if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
          {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
          }
      }

    if (lane < bs)
      {
        y[block_row * bs + lane] = partial_sums[threadIdx.x];
      }
  },
    (const int *, col_ids),
    (const int *, row_ptr),
    (const float *, data),
    (const float *, x),
    (float*, y));
  nlohmann::json json;
  json["bs"] = bs;
  json["stride_begin"] = round_up_to_power_of_two((32 / bs) / 2);
  json["shared_size"] = block_size.x;
  auto bcsr_kernel = bcsr_jit.compile (json);

  const index_type matrix_size =
      num_blocks * bs * bs;
  const index_type columns_size = num_blocks;
  const index_type row_ptr_size = n + 1;
  const index_type x_size = rows;
  const index_type y_size = rows;


  data_type *cu_x;
  data_type *cu_y;
  CudaCheck(cudaMalloc(&cu_x, x_size * sizeof(data_type)));
  CudaCheck(cudaMemcpy(cu_x, x, x_size * sizeof(data_type), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&cu_y, y_size * sizeof(data_type)));
  CudaCheck(cudaMemcpy(cu_y, y, y_size * sizeof(data_type), cudaMemcpyHostToDevice));

  data_type *val_d;
  index_type *col_ind_d, *block_ptr_d;
  cudaMalloc(&val_d, num_blocks * bs * bs * sizeof(data_type));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(index_type));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(index_type));
  cudaMemcpy(val_d, val_t, num_blocks * bs * bs * sizeof(data_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(index_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(index_type),
             cudaMemcpyHostToDevice);




  bcsr_kernel.launch(grid_size, block_size, col_ind_d, block_ptr_d, val_d, cu_x,
                     cu_y);


  cudaMemcpy(y, cu_y, y_size * sizeof(data_type), cudaMemcpyDeviceToHost);
  //cudaFree(d_values);
  //cudaFree(d_x);
  //cudaFree(d_y);
  //cudaFree(d_row_ptr);
  //cudaFree(d_columns);


}

void test_spmv(float *val, int *block_ptr, int *col_ind, float *x, int n,
               int blockSize, int num_blocks, int rows) {
  // float *ysparse=(float* )malloc(sizeof(float)*rows);
  // spmv_cusparse(val,block_ptr,col_ind,x,ysparse,n,blockSize,num_blocks,rows);
  // for(int i=0;i<rows;i++){
  //  printf("%.1f ",ysparse[i]);
  //}
  // printf("\n" );
  float *yserial = (float *)malloc(sizeof(float) * rows);
  for (int i = 0; i < rows; i++) {
    printf("%5d", i);
  }
  printf("\n");
  for (int i = 0; i < rows; i++) {
    yserial[i] = 0;
  }
  serial(val, block_ptr, col_ind, x, yserial, n, blockSize);
  for (int i = 0; i < rows; i++) {
    printf("%5.1f", yserial[i]);
  }
  printf("\n");
  float *ybsr1 = (float *)malloc(sizeof(float) * rows);
  for (int i = 0; i < rows; i++) {
    ybsr1[i] = 0;
  }
  bsr1run(val, block_ptr, col_ind, x, ybsr1, n, blockSize, num_blocks, rows);
  for (int i = 0; i < rows; i++) {
    printf("%5.1f", ybsr1[i]);
  }
  printf("\n");
  float *ybsr2 = (float *)malloc(sizeof(float) * rows);
  for (int i = 0; i < rows; i++) {
    ybsr2[i] = 0;
  }
  bsr2run(val, block_ptr, col_ind, x, ybsr2, n, blockSize, num_blocks, rows);
  for (int i = 0; i < rows; i++) {
    printf("%5.1f", ybsr2[i]);
  }
  printf("\n");
  float *yjit = (float *)malloc(sizeof(float) * rows);
  for (int i = 0; i < rows; i++) {
    yjit[i] = 0;
  }
  spmv_jit(val, block_ptr, col_ind,x, yjit, n, blockSize,  num_blocks,  rows);
  for (int i = 0; i < rows; i++) {
    printf("%5.1f", yjit[i]);
  }

}

int main(int argc, char **argv) {

  ofstream myfile;
  myfile.open("matrix.txt");
  int N = 1 << atoi(argv[1]);
  int K = N;
  int bs = 1 << atoi(argv[2]);
  myfile << "N= " << N << " bs= " << bs << "\n";
  float density = (float)1 / (1 << (atoi(argv[3])));
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

  float *x = (float *)malloc(rows * sizeof(float));
  for (int i = 0; i < rows; i++) {
    x[i] = 1;
  }
  test_spmv(weight, weight_ptr, weight_ind, x, N, bs, num_blocks, rows);
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
  int nnzcntr = 0;
  for (int i = 0; i < N; i++) {
    int block_first = weight_ptr[i];
    int block_last = weight_ptr[i + 1];
    for (int block = block_first; block < block_last; block++) {
      for (int row = 0; row < bs; row++) {
        for (int col = 0; col < bs; col++) {
          myfile << i * bs + row << " " << weight_ind[block] * bs + col << " "
                 << weight[block * bs * bs + row * bs + col] << "\n";

          nnzcntr++;
          // bsr[ i * bs + row][weight_ind[block] * bs + col]=weight[block * bs
          // * bs + row * bs + col];
        }
      }
    }
  }
  printf("nnzcntr=%d\n", nnzcntr);
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
  myfile.close();
  return 0;
}
