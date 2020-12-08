#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
using namespace std;
float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
  return dis(e);
}

int get_random_int(int max) {
  static std::default_random_engine e;
  std::uniform_int_distribution<int> dis(0, max);
  return dis(e);
}

void generate_candidate_blocks(int R, int C, int BS_R, int BS_C, int num_blocks,
                               int *weight_indptr, int *weight_indices) {
  std::map<int, std::set<int>> blocks;
  int num_r_block = R / BS_R;
  int num_c_block = C / BS_C;
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

void compute_BSR(float *val, float *block_ptr, float *col_ind, float *x,
                 float *y, int n, int blockSize, int num_blocks){
  // For blas 2 gemv y = alpha.x.A + Beta.y
  const float alpha = 1.0;
  const float beta = 0.0;
  // Copy input
  float *cu_x;
  cudaMalloc(&cu_x, bsr1run * sizeof(float));
  cudaMemcpy(cu_x, x, bsr1run * sizeof(float), cudaMemcpyHostToDevice);

  float *cu_y;
  cudaMalloc(&cu_x, n * sizeof(float));

  // Init matrix properties
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descr = 0;
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv

  cusparseDbsrmv(cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                 CUSPARSE_OPERATION_NON_TRANSPOSE, n, n, num_blocks, &alpha,
                 descr, val, block_ptr, col_ind, blockSize, cu_x, &beta, cu_y);

  // Get back result
  cudaMemcpy(y, cu_y, n * sizeof(double), cudaMemcpyDeviceToHost);
  // Dealloc vectors
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));
}

void serial(float *val, float *block_ptr, float *col_ind, float *x,
                 float *y, int n, int bs){
for (int i = 0; i < n; i++) {
  int block_first = block_ptr[i];
  int block_last = block_ptr[i + 1];
  for (int block = block_first; block < block_last; block++) {
    for (int row = 0; row < bs; row++) {
      for (int col = 0; col < bs; col++) {
        //printf("%d  %d  %f \n", i * bs + row,
               int row_v=i*bs+row;
               int column =col_ind[block] * bs + col;
               y[row_v]+=val[block * bs * bs + row * bs + col]*x[column];
      }
    }
  }
}
}
template <typename data_type, typename index_type>
__global__ void bsr1 (
  index_type n_block_rows,
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type *y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = idx % bs;
  const index_type block_row = idx / bs;
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  if (row < bs && block_row < n_block_rows)
    {
      data_type local_out = 0.0;

      for (index_type block = first_block; block < last_block; block++)
        {
          const index_type first_col = col_ids[block] * bs;
          for (index_type col = 0; col < bs; col++)
            local_out += x[first_col + col] * data[block * bs * bs + row * bs + col];
        }

      y[block_row * bs + row] = local_out;
    }
}

void bsr1run(float *val, float *block_ptr, float *col_ind, float *x, int n,
               int blockSize, int num_blocks,int rows) {
  double *cu_x;
  double *cu_y;
  CudaCheck(cudaMalloc(&cu_x, rows * sizeof(double)));
  CudaCheck(cudaMemcpy(cu_x, x, rows * sizeof(double), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&cu_y, rows * sizeof(double)));
  dim3 block_size = 512;
  dim3 grid_size {};

  grid_size.x = (n * blockSize + block_size.x - 1) / block_size.x;

  //bcsr_spmv_kernel_column_by_column_template<<<grid_size, block_size, block_size.x * sizeof(double)>>>(bcsr.cu_bsrColIndC,bcsr.cu_bsrRowPtrC,bcsr.cu_bsrValC,cu_x,cu_y);
  bsr1<float, int><<<grid_size, block_size>>>(n, blockSize, colind, block_ptr, val, cu_x, cu_y);



  // Get back result
  //

  cudaMemcpy(y, cu_y, rows * sizeof(double), cudaMemcpyDeviceToHost);
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));


}
void test_spmv(float *val, float *block_ptr, float *col_ind, float *x, int n,
               int blockSize, int num_blocks,int rows) {
//float *ysparse=(float* )malloc(sizeof(float)*rows);
//compute_BSR(val,block_ptr,col_ind,x,y,n,block_size,num_blocks);
float *yserial=(float* )malloc(sizeof(float)*rows);
serial(val,block_ptr,col_ind,x,yserial,n,blockSize);
//float* ybsr1=(float *)malloc(sizeof(float)*rows);
//bsr1run(val,block_ptr,col_ind,x,y,n,block_size,num_blocks,bsr1run);
               }

int main(int argc, char **argv) {
  int M = 1 << atoi(argv[1]);
  int N = 1 << atoi(argv[2]);
  int K = N;
  int BS_R = 1 << atoi(argv[3]);
  int BS_C = BS_R;
  float density = 0.20;

  float *data;
  float *weight;
  int *weight_ind;
  int *weight_ptr;
  int nnz = int(density * M * N);
  int num_blocks = int(nnz / (BS_R * BS_C)) + 1;

  data = (float *)malloc(M * K * sizeof(float));
  weight = (float *)malloc(num_blocks * BS_R * BS_R * sizeof(float));
  weight_ind = (int *)malloc(num_blocks * sizeof(int));
  weight_ptr = (int *)malloc((N + 1) * sizeof(int));

  for (int i = 0; i < M * K; i++) {
    data[i] = get_random();
  }

  for (int i = 0; i < num_blocks * BS_R * BS_C; i++) {
    weight[i] = get_random();
  }

  generate_candidate_blocks(N, K, BS_R, BS_R, num_blocks, weight_ptr,
                            weight_ind);
  int rows=N*BS_R;

  float *x = malloc(N * sizeof(float));
  for(int i=0;i<n;i++){x[i]=1;}
    test_spmv( weight, weight_ptr, weight_ind, x, n,
                 BS_R, num_blocks, rows);
                   /*
    for (int i = 0; i < N; i++) {
      int block_first = weight_ptr[i];
      int block_last = weight_ptr[i + 1];
      for (int block = block_first; block < block_last; block++) {
        for (int row = 0; row < BS_R; row++) {
          for (int col = 0; col < BS_R; col++) {
            printf("%d  %d  %f \n", i * BS_R + row,
                   weight_ind[block] * BS_R + col,
                   weight[block * BS_R * BS_R + row * BS_R + col]);
          }
        }
      }
    }
  */
  return 0;
}
