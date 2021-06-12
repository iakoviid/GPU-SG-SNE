/*!
  \file   graph_rescaling.cpp
  \brief  Routines regarding lambda-based graph rescaling.

*/
#include "graph_rescaling.cuh"

#define Blocksize 512
/*we will use warps for the parallel evaluation of the expression and the serial
 * code for changing the interval*/
__global__ void bisectionSearchKernel(volatile matval *__restrict__ sig2,
                                      volatile matval *__restrict__ p_sp,
                                      const matidx *const ir, const int n,
                                      const matval lambda,
                                      const matval tolerance,
                                      const bool dropLeafEdge) {
  __shared__ matval sdata[Blocksize / 32];
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int warp_id_block = threadIdx.x / 32;
  register const unsigned int lane = thread_id % 32;
  register unsigned int row = warp_id;
  const unsigned int n_warps = gridDim.x * blockDim.x / 32;
  register matval sigma;
  register matval a;
  register matval c;
  register matval sum;
  register matval perplexity_diff;
  for (; row < n; row = row + n_warps) {
    a = -1e3;
    c = 1e7;
    sum = 0;
    sigma = 1;
    perplexity_diff = 1 - lambda;
    int found = 0;
    int iter = 0;
    unsigned int row_start = ir[row];
    unsigned int row_end = ir[row + 1];
    while (__all_sync(FULL_WARP_MASK, found != 1) && iter < 100) {
      sum = 0;
      for (unsigned int element = row_start + lane; element < row_end;
           element += 32) {
        sum += expf(-p_sp[element] * sigma);
      }
      sum = warp_reduce(sum);
      if (lane == 0) {
        perplexity_diff = sum - lambda;
        if (perplexity_diff < tolerance && perplexity_diff > -tolerance) {
          found = 1;
        }
        if (perplexity_diff > 0) {
          a = sigma;
          if (c > 1e7) {
            sigma = 2 * a;
          } else {
            sigma = 0.5 * (a + c);
          }

        } else {
          c = sigma;
          sigma = 0.5 * (a + c);
        }
        sdata[warp_id_block] = sigma;
      }
      __syncwarp(FULL_WARP_MASK);
      sigma = sdata[warp_id_block];
      iter++;
    }
    if (lane == 0) {
      sig2[row] = sigma;
    }
    sum = 0;
    for (unsigned int element = row_start + lane; element < row_end;
         element += 32) {
      p_sp[element] = expf(-p_sp[element] * sigma);
      sum += p_sp[element];
    }
    sum = warp_reduce(sum);
    if (lane == 0) {
      sdata[warp_id_block] = sum;
    }
    __syncwarp(FULL_WARP_MASK);

    sum = sdata[warp_id_block];
    for (unsigned int element = row_start + lane; element < row_end;
         element += 32) {
      p_sp[element] /= sum;
    }

    // override lambda value of leaf node?
    if (dropLeafEdge && (row_end - row_start == 1))
      p_sp[row_start] = 0;
  }
}

void lambdaRescalingGPU(sparse_matrix<matval> P, matval lambda, bool dist,
                        bool dropLeafEdge) {
  matval tolBinary = 1e-5;
  // int    maxIter     = 100;
  thrust::device_vector<matval> sig2(P.n);
  if (dist)
    std::cout << "Input considered as distances" << std::endl;

  bisectionSearchKernel<<<64, Blocksize>>>(
      thrust::raw_pointer_cast(sig2.data()), P.val, P.row, P.n, lambda,
      tolBinary, dropLeafEdge);
}
#define N_THREADS 1024
__global__ void makeStochasticKernel(matval *val, matidx *row, uint32_t n,
                                     uint32_t *stoch) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = thread_id / 32;
  const uint32_t lane = thread_id % 32;
  __shared__ coord sdata[N_THREADS / 32];
  const unsigned int warp_id_block = threadIdx.x / 32;

  const unsigned int n_warps = gridDim.x * blockDim.x / 32;
  for (uint32_t j = warp_id; j < n; j = j + n_warps) {
    matval sum = 0;
    for (uint32_t t = row[j] + lane; t < row[j + 1]; t += 32) {
      sum += val[t];
    }
    sum = warp_reduce(sum);
    if (lane == 0) {
      sdata[warp_id_block] = sum;
    }
    __syncwarp(FULL_WARP_MASK);
    sum = sdata[warp_id_block];

    if (fabs(sum - 1) > 1e-5) {
      for (uint32_t t = row[j] + lane; t < row[j + 1]; t += 32) {
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
uint32_t makeStochasticGPU(coord *val, int *row, int n) {

  uint32_t *stoch;
  CUDA_CALL(cudaMallocManaged(&stoch, n * sizeof(uint32_t)));

  makeStochasticKernel<<<64, N_THREADS>>>(val, row, n, stoch);
  cudaDeviceSynchronize();

  uint32_t nStoch = thrust::reduce(stoch, stoch + n);

  CUDA_CALL(cudaFree(stoch));
  return nStoch;
}

uint32_t makeStochasticGPU(sparse_matrix<matval> *P) {

  uint32_t *stoch;
  CUDA_CALL(cudaMallocManaged(&stoch, P->n * sizeof(uint32_t)));

  makeStochasticKernel<<<64, 512>>>(P->val, P->row, P->n, stoch);
  cudaDeviceSynchronize();

  uint32_t nStoch = thrust::reduce(stoch, stoch + P->n);

  CUDA_CALL(cudaFree(stoch));
  return nStoch;
}
/* (P+P^T)/2*/
sparse_matrix<matval>* symmetrizeMatrixGPU(sparse_matrix<matval> *A, cusparseHandle_t &handle) {
  // Sort the matrix properly
  size_t permutation_buffer_byte_size = 0;
  void *permutation_buffer = NULL;
  int32_t *permutation = NULL;

  // Initialize the matrix descriptor
  cusparseMatDescr_t matrix_descriptor;
  cusparseCreateMatDescr(&matrix_descriptor);
  cusparseSetMatType(matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

  // step 1: Allocate memory buffer
  cusparseXcsrsort_bufferSizeExt(handle, A->m, A->n, A->nnz, A->row, A->col,
                                 &permutation_buffer_byte_size);
  cudaDeviceSynchronize();
  cudaMalloc(&permutation_buffer, sizeof(char) * permutation_buffer_byte_size);

  // step 2: Setup permutation vector permutation to be the identity
  cudaMalloc(reinterpret_cast<void **>(&permutation), sizeof(int32_t) * A->nnz);
  cusparseCreateIdentityPermutation(handle, A->nnz, permutation);
  cudaDeviceSynchronize();

  // step 3: Sort CSR format
  cusparseXcsrsort(handle, A->m, A->n, A->nnz, matrix_descriptor, A->row,
                   A->col, permutation, permutation_buffer);
  cudaDeviceSynchronize();

  // step 4: Gather sorted csr_values
  float *csr_values_a_sorted = nullptr;
  cudaMalloc(reinterpret_cast<void **>(&csr_values_a_sorted),
             (A->nnz) * sizeof(float));
  cusparseSgthr(handle, A->nnz, A->val, csr_values_a_sorted, permutation,
                CUSPARSE_INDEX_BASE_ZERO);
  cudaDeviceSynchronize();

  // Free some memory
  cudaFree(permutation_buffer);
  cudaFree(permutation);
  A->val = csr_values_a_sorted;

  coord *d_csrValB;
  CUDA_CALL(cudaMalloc(&d_csrValB, A->nnz * sizeof(coord)));
  int *d_csrRowPtrB;
  CUDA_CALL(cudaMalloc(&d_csrRowPtrB, (A->m + 1) * sizeof(int)));
  int *d_csrColIndB;
  CUDA_CALL(cudaMalloc(&d_csrColIndB, A->nnz * sizeof(int)));

  cusparseScsr2csc(handle, A->m, A->n, A->nnz, A->val, A->row, A->col,
                   d_csrValB, d_csrColIndB, d_csrRowPtrB,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  cudaDeviceSynchronize();
  // --- Summing the two matrices
  int baseC, nnz3;
  coord *sym_val;
  int *sym_col, *sym_row;
  // --- nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnz3;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  CUDA_CALL(cudaMalloc(&sym_row, (A->m + 1) * sizeof(int)));
  cusparseXcsrgeamNnz(handle, A->m, A->n, matrix_descriptor, A->nnz, A->row,
                      A->col, matrix_descriptor, A->nnz, d_csrRowPtrB,
                      d_csrColIndB, matrix_descriptor, sym_row,
                      nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
    nnz3 = *nnzTotalDevHostPtr;
  } else {
    CUDA_CALL(
        cudaMemcpy(&nnz3, sym_row + A->m, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CALL(cudaMemcpy(&baseC, sym_row, sizeof(int), cudaMemcpyDeviceToHost));
    nnz3 -= baseC;
  }

  CUDA_CALL(cudaMalloc(&sym_col, nnz3 * sizeof(int)));
  CUDA_CALL(cudaMalloc(&sym_val, nnz3 * sizeof(coord)));
  coord alpha = 0.5, beta = 0.5;
  cusparseScsrgeam(handle, A->m, A->n, &alpha, matrix_descriptor, A->nnz,
                   A->val, A->row, A->col, &beta, matrix_descriptor, A->nnz,
                   d_csrValB, d_csrRowPtrB, d_csrColIndB, matrix_descriptor,
                   sym_val, sym_row, sym_col);
  cudaDeviceSynchronize();

  CUDA_CALL(cudaFree(d_csrValB));
  CUDA_CALL(cudaFree(d_csrRowPtrB));
  CUDA_CALL(cudaFree(d_csrColIndB));
  sparse_matrix<coord>* C=(sparse_matrix<coord> *)malloc(sizeof(sparse_matrix<coord>));
  C->n=A->n;
  C->m=A->m;
  C->nnz = nnz3;
  C->row = sym_row;
  C->col = sym_col;
  C->val = sym_val;
  return C;
}
int SymmetrizeMatrix(cusparseHandle_t &handle,
        float** d_symmetrized_values,
        int** d_symmetrized_rowptr,
        int** d_symmetrized_colind,
        float *csr_values_a,
        int *csr_column_ptr_a,
        const int num_points,
        int* csr_row_ptr_a,
        const int nnz
      )
{

    // Initialize the matrix descriptor
    cusparseMatDescr_t matrix_descriptor;
    cusparseCreateMatDescr(&matrix_descriptor);
    cusparseSetMatType(matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

    // Sort the matrix properly
    size_t permutation_buffer_byte_size = 0;
    void *permutation_buffer = NULL;
    int32_t *permutation = NULL;

    // step 1: Allocate memory buffer
    cusparseXcsrsort_bufferSizeExt(handle, num_points, num_points,
            nnz, csr_row_ptr_a,
            csr_column_ptr_a, &permutation_buffer_byte_size);
    cudaDeviceSynchronize();
    cudaMalloc(&permutation_buffer,
               sizeof(char)*permutation_buffer_byte_size);

    // step 2: Setup permutation vector permutation to be the identity
    cudaMalloc(reinterpret_cast<void**>(&permutation),
            sizeof(int32_t)*nnz);
    cusparseCreateIdentityPermutation(handle, nnz,
                                      permutation);
    cudaDeviceSynchronize();

    // step 3: Sort CSR format
    cusparseXcsrsort(handle, num_points, num_points,
            nnz, matrix_descriptor, csr_row_ptr_a,
            csr_column_ptr_a, permutation, permutation_buffer);
    cudaDeviceSynchronize();

    // step 4: Gather sorted csr_values
    float* csr_values_a_sorted = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csr_values_a_sorted),
            (nnz)*sizeof(float));
    cusparseSgthr(handle,nnz, csr_values_a,
            csr_values_a_sorted, permutation, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Free some memory
    cudaFree(permutation_buffer);
    cudaFree(permutation);
    csr_values_a = csr_values_a_sorted;

    // We need A^T, so we do a csr2csc() call
    int32_t* csc_row_ptr_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_row_ptr_at),
            (nnz)*sizeof(int32_t));
    int32_t* csc_column_ptr_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_column_ptr_at),
            (num_points+1)*sizeof(int32_t));
    float* csc_values_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_values_at),
            (nnz)*sizeof(float));

    // Do the transpose operation
    cusparseScsr2csc(handle, num_points, num_points,
                     nnz, csr_values_a, csr_row_ptr_a,
                     csr_column_ptr_a, csc_values_at, csc_row_ptr_at,
                     csc_column_ptr_at, CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Now compute the output size of the matrix
    int32_t base_C, num_nonzeros_C;
    int32_t symmetrized_num_nonzeros = -1;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    //d_symmetrized_rowptr.resize(num_points+1);
    CUDA_CALL(cudaMallocManaged(&(*d_symmetrized_rowptr), (num_points+1) * sizeof(int)));

    cusparseXcsrgeamNnz(handle, num_points, num_points,
            matrix_descriptor, nnz, csr_row_ptr_a,
                csr_column_ptr_a,
            matrix_descriptor, nnz, csc_column_ptr_at,
                csc_row_ptr_at,
            matrix_descriptor,
            (*d_symmetrized_rowptr),
            &symmetrized_num_nonzeros);
    cudaDeviceSynchronize();

    // Do some useful checking...
    if (-1 != symmetrized_num_nonzeros) {
        num_nonzeros_C = symmetrized_num_nonzeros;
    } else {
        cudaMemcpy(&num_nonzeros_C,
                (*d_symmetrized_rowptr) +
                num_points, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base_C,
                (*d_symmetrized_rowptr),
                sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Allocate memory for the new summed array
    CUDA_CALL(cudaMallocManaged(&(*d_symmetrized_colind), num_nonzeros_C * sizeof(int)));
    CUDA_CALL(cudaMallocManaged(&(*d_symmetrized_values), num_nonzeros_C * sizeof(float)));

    // Sum the arrays
    //float kAlpha = 1.0f ;
    //float kBeta = 1.0f ;
    float kAlpha = 1.0f / (2.0f * num_points);
    float kBeta = 1.0f / (2.0f * num_points);
    cusparseScsrgeam(handle, num_points, num_points,
            &kAlpha, matrix_descriptor, nnz,
            csr_values_a, csr_row_ptr_a, csr_column_ptr_a,
            &kBeta, matrix_descriptor, nnz,
            csc_values_at, csc_column_ptr_at, csc_row_ptr_at,
            matrix_descriptor,
            (*d_symmetrized_values),
            (*d_symmetrized_rowptr),
            (*d_symmetrized_colind));
    cudaDeviceSynchronize();

    // Free the memory we were using...
    cudaFree(csr_values_a);
    cudaFree(csc_values_at);
    cudaFree(csr_row_ptr_a);
    cudaFree(csc_column_ptr_at);
    cudaFree(csc_row_ptr_at);
    return num_nonzeros_C;
}
