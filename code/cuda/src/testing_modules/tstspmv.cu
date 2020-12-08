#include <cuda_runtime.h>
#include <cusparse.h>

#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
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
void maxError(data_type* yref,data_type* y,index_type n){
  data_type maxError=0;
  for(int i=0;i<n;i++){
    data_type Error= (yref[i]-y[i])*(yref[i]-y[i]);
    if (Error>maxError){

      maxError=Error;
    }
  }
  cout<<"maxError="<<maxError<<"\n";

}
void test_bsr(float *val, int *block_ptr, int *col_ind, float *x, int n,
               int blockSize, int num_blocks, int rows) {
/*
   float *ysparse=(float* )malloc(sizeof(float)*rows);
   spmv_cusparse(val,block_ptr,col_ind,x,ysparse,n,blockSize,num_blocks,rows);
   for(int i=0;i<rows;i++){
    printf("%5.1f",ysparse[i]);
  }
   printf("\n" );


  */
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

  maxError(yserial,ybsr1,rows);
  maxError(yserial,ybsr2,rows);

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
  cudaMemcpy(csrRptr,csrRowPtrC,rows*sizeof(int),cudaMemcpyDeviceToHost);



}
template <typename data_type, typename index_type>
void csrspmvcusparse(data_type* val,index_type* col,index_type* rptr,index_type n,index_type nnz,data_type* x, data_type* y){

  data_type *cu_x;
  data_type *cu_y;
  CudaCheck(cudaMalloc(&cu_x, n * sizeof(data_type)));
  CudaCheck(cudaMemcpy(cu_x, x, n * sizeof(data_type), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&cu_y, n * sizeof(data_type)));
  CudaCheck(cudaMemcpy(cu_y, y, n * sizeof(data_type), cudaMemcpyHostToDevice));
  const data_type alpha = 1.0;
  const data_type beta = 0.0;

  data_type* val_d;
  index_type* col_d;
  index_type* rptr_d;
  CudaCheck(cudaMalloc(&val_d, nnz * sizeof(data_type)));
  CudaCheck(cudaMemcpy(val_d, val,nnz * sizeof(data_type), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&col_d, nnz * sizeof(index_type)));
  CudaCheck(cudaMemcpy(col_d, col,nnz * sizeof(index_type), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&rptr_d, nnz * sizeof(index_type)));
  CudaCheck(cudaMemcpy(rptr_d, rptr,n * sizeof(index_type), cudaMemcpyHostToDevice));

  cusparseMatDescr_t descr = 0;
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseScsrmv(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, n, n,
      nnz, &alpha, descr, val_d, rptr_d,
      col_d, cu_x, &beta, cu_y);


  cudaMemcpy(y, cu_y, n * sizeof(data_type), cudaMemcpyDeviceToHost);



}


template <typename data_type, typename index_type>
void test_csr(data_type* val,index_type* col,index_type* rptr,index_type n,index_type nnz,data_type* x){

  data_type* ycusparse=(data_type *)malloc(sizeof(data_type*)*n);
  csrspmvcusparse(val,col,rptr,n,nnz,x,ycusparse);


}
template <typename data_type, typename index_type>
void test_coo(data_type* coov,index_type* cooi,index_type* cooj,index_type nnz,data_type* x){

  
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
  test_bsr(weight, weight_ptr, weight_ind, x, N, bs, num_blocks, rows);
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

  float* csrVal=(float *)malloc(sizeof(float)*num_blocks*bs*bs);
  int* csrCidx=(int *)malloc(sizeof(int)*num_blocks*bs*bs);
  int* csrRptr=(int *)malloc(sizeof(int)*rows);

  bsr2csr(weight, weight_ptr, weight_ind, N, bs, num_blocks, rows, csrVal, csrRptr,csrCidx);

  nnz=num_blocks*bs*bs;
  test_csr(csrVal,csrCidx,csrRptr,rows,nnz,x);

  test_coo(coov,cooi,cooj,nnz,x);

  //for(int i=0;i<rows;i++){printf("%d ",csrRptr[i] );}


  printf("nnz=%d vs nnz=%d\n", nnzcntr,nnz);
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
