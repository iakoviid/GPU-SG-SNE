#include "sparsematrix.cuh"

/*

void free_sparse_matrix(sparse_matrix * P){
  free(P->row);
  free(P->col);
  free(P->val);

}
*/
void free_sparse_matrixGPU(sparse_matrix *P) {

  CUDA_CALL(cudaFree(P->row));
  CUDA_CALL(cudaFree(P->col));
  CUDA_CALL(cudaFree(P->val));
}

__global__ void makeStochasticKernel(matval *val, matidx *col, matidx *row,
                                     uint32_t n, uint32_t *stoch) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = thread_id / 32;
  const uint32_t lane = thread_id % 32;
  __shared__ coord sdata[256 / 32];
  const unsigned int warp_id_block = threadIdx.x / 32;

  const unsigned int n_warps=gridDim.x*blockDim.x/32;
  for (uint32_t j=warp_id;j < n;j=j+n_warps) {
    matval sum = 0;
    for (uint32_t t = row[j]+lane; t < row[j + 1]; t += 32) {
      sum += val[t];
    }
    sum=warp_reduce(sum);
    if (lane == 0) {
      sdata[warp_id_block] = sum;
    }
    __syncwarp(FULL_WARP_MASK);
    sum = sdata[warp_id_block];

    if (fabs(sum - 1) > 1e-12) {
      for (uint32_t t = row[j]+lane; t < row[j + 1]; t += 32) {
        val[t] /= sum;
      }
      if (lane == 0) {
        stoch[j] = 0;
      }
    } else {
      if (lane == 0) {
        stoch[j] = 1;
      }
    }
  }
}

uint32_t makeStochasticGPU(sparse_matrix *P) {

  uint32_t *stoch;
  CUDA_CALL(cudaMallocManaged(&stoch, P->n * sizeof(uint32_t)));

  makeStochasticKernel<<<256, 32>>>(P->val, P->col, P->row, P->n, stoch);

  uint32_t nStoch = thrust::reduce(stoch, stoch + P->n);

  CUDA_CALL(cudaFree(stoch));
  return nStoch;
}

void permuteMatrixGPU(sparse_matrix *P, int *perm, int *iperm) {
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  // Get sparse matrix
  matidx *row_P = P->row;
  matidx *col_P = P->col;
  matval *val_P = P->val;

  int N = P->n;
  matidx nnz = P->nnz;

  // Allocate memory for permuted matrix
  matval *perm_val_P = (matval *)malloc(nnz * sizeof(matval));
  CUDA_CALL(cudaMallocManaged(&perm_val_P, nnz * sizeof(matval)));

  size_t pBufferSizeInBytes = 0;
  void *pBuffer = NULL;

  // step 1: allocate buffer
  cusparseXcsrsort_bufferSizeExt(handle, N, N, nnz, col_P, row_P,
                                 &pBufferSizeInBytes);
  cudaMalloc(&pBuffer, sizeof(char) * pBufferSizeInBytes);

  // step 3: sort CSR format
  cusparseXcsrsort(handle, N, N, nnz, descrA, col_P, row_P, perm, pBuffer);

  cusparseDgthr(handle, nnz, val_P, perm_val_P, perm, CUSPARSE_INDEX_BASE_ZERO);

  cudaFree(P->val);
  P->val = perm_val_P;
}
void add_cusparse(sparse_matrix A, sparse_matrix B, sparse_matrix *C,
                  cusparseHandle_t handle) {
  // --- Initialize matrix descriptors
  cusparseMatDescr_t descrA, descrB, descrC;
  cusparseCreateMatDescr(&descrA);
  cusparseCreateMatDescr(&descrB);
  cusparseCreateMatDescr(&descrC);
  int M = A.m;
  int N = A.n;
  // --- Summing the two matrices
  int baseC, nnz3;
  // --- nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnz3;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  int *d_csrRowPtrC;
  cudaMalloc(&d_csrRowPtrC, (M + 1) * sizeof(int));
  cusparseXcsrgeamNnz(handle, M, N, descrA, A.nnz, A.row, A.col, descrB, B.nnz,
                      B.row, B.col, descrC, d_csrRowPtrC, nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
    nnz3 = *nnzTotalDevHostPtr;
  } else {
    cudaMemcpy(&nnz3, d_csrRowPtrC + M, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, d_csrRowPtrC, sizeof(int), cudaMemcpyDeviceToHost);
    nnz3 -= baseC;
  }
  int *d_csrColIndC;
  cudaMalloc(&d_csrColIndC, nnz3 * sizeof(int));
  matval *d_csrValC;
  cudaMalloc(&d_csrValC, nnz3 * sizeof(matval));
  matval alpha = (matval)1 / 2, beta = (matval)1 / 2;
  cusparseDcsrgeam(handle, M, N, &alpha, descrA, A.nnz, A.val, A.row, A.col,
                   &beta, descrB, B.nnz, B.val, B.row, B.col, descrC, d_csrValC,
                   d_csrRowPtrC, d_csrColIndC);

  C->row = d_csrRowPtrC;
  C->nnz = nnz3;
  C->col = d_csrColIndC;
  C->val = d_csrValC;
  C->m = M;
  C->n = N;
}

void symmetrizeMatrixGPU(sparse_matrix *A, sparse_matrix *C) {
  int M = A->m;
  int N = A->n;
  int nnz = A->nnz;
  int *csc_row_ptr_at;
  cudaMalloc(reinterpret_cast<void **>(&csc_row_ptr_at),
             (A->nnz) * sizeof(int));
  int *csc_column_ptr_at;
  cudaMalloc(reinterpret_cast<void **>(&csc_column_ptr_at),
             (A->n + 1) * sizeof(int));
  matval *csc_values_at;
  cudaMalloc(reinterpret_cast<void **>(&csc_values_at),
             (A->nnz) * sizeof(matval));
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  // Do the transpose operation
  printf("nnz=%d n=%d\n",A->nnz,A->n );
  cusparseDcsr2csc(handle, A->m, A->n, A->nnz, A->val, A->row, A->col,
                   csc_values_at, csc_row_ptr_at, csc_column_ptr_at,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

  cudaDeviceSynchronize();

  cusparseMatDescr_t descrA, descrB, descrC;
  cusparseCreateMatDescr(&descrA);
  cusparseCreateMatDescr(&descrB);
  cusparseCreateMatDescr(&descrC);
  // --- Summing the two matrices
  int baseC, nnz3;
  // --- nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnz3;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cudaMalloc(&C->row, (M + 1) * sizeof(int));
  cusparseXcsrgeamNnz(handle, M, N, descrA, A->nnz, A->row, A->col, descrB, nnz,
                      csc_column_ptr_at, csc_row_ptr_at, descrC, C->row,
                      nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
    nnz3 = *nnzTotalDevHostPtr;
  } else {
    cudaMemcpy(&nnz3, C->row + M, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, C->row, sizeof(int), cudaMemcpyDeviceToHost);
    nnz3 -= baseC;
  }
  cudaMalloc(&C->col, nnz3 * sizeof(int));
  cudaMalloc(&C->val, nnz3 * sizeof(matval));
  matval alpha = (matval)1 / 2, beta = (matval)1 / 2;
  cusparseDcsrgeam(handle, M, N, &alpha, descrA, nnz, A->val, A->row, A->col,
                   &beta, descrB, nnz, csc_values_at, csc_column_ptr_at,
                   csc_row_ptr_at, descrC, C->val, C->row, C->col);

  C->nnz = nnz3;
  C->m = M;
  C->n = N;
  cudaDeviceSynchronize();
  printf("nnz=%d n=%d \n",C->nnz,C->n );

  cudaFree(csc_column_ptr_at);
  cudaFree(csc_row_ptr_at);
  cudaFree(csc_values_at);
}
