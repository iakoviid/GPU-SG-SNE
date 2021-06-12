#include "prepareMatrix.cuh"
#include "sparse_reorder.cuh"
__global__ void Csr2CooKernel(volatile int *__restrict__ coo_indices,
                              const int *__restrict__ pij_row_ptr,
                              const int *__restrict__ pij_col_ind,
                              const int num_points, const int num_nonzero) {
  register int TID, i, j, start, end;
  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_nonzero)
    return;
  start = 0;
  end = num_points + 1;
  i = (num_points + 1) >> 1;
  while (end - start > 1) {
    j = pij_row_ptr[i];
    end = (j > TID) ? i : end;
    start = (j <= TID) ? i : start;
    i = (start + end) >> 1;
  }
  coo_indices[TID] = i;
}
void Csr2Coo(int nnz, int n, int *row, int *col, int *coo_indices) {
  const int num_threads = 1024;
  const int num_blocks = iDivUp(nnz, num_threads);

  Csr2CooKernel<<<num_blocks, num_threads>>>(coo_indices, row, col, n, nnz);
  cudaDeviceSynchronize();
}

void Csr2Coo(sparse_matrix<float> *P) {
  const int num_threads = 1024;
  const int num_blocks = iDivUp(P->nnz, num_threads);
  int *coo_indices;
  cudaMalloc((void **)&coo_indices, sizeof(int) * (P->nnz));

  Csr2CooKernel<<<num_blocks, num_threads>>>(coo_indices, P->row, P->col, P->n,
                                             P->nnz);
  cudaDeviceSynchronize();
  cudaFree(P->row);
  P->row = coo_indices;
}
void Csr2Coo(sparse_matrix<float> **P) {
  const int num_threads = 1024;
  const int num_blocks = iDivUp((*P)->nnz, num_threads);
  int *coo_indices;
  cudaMalloc((void **)&coo_indices, sizeof(int) * ((*P)->nnz));

  Csr2CooKernel<<<num_blocks, num_threads>>>(coo_indices, (*P)->row, (*P)->col,
                                             (*P)->n, (*P)->nnz);
  cudaDeviceSynchronize();
  cudaFree((*P)->row);
  (*P)->row = coo_indices;
}
void csr2bsr(int blockDim, int n, int m, int nnz, int *csrRowptr,
             int *csrColInd, float *csrVal, int **bsrRowPtr, int **bsrColInd,
             float **bsrVal, int *nnzblocks, int *n_block_rows,
             cusparseHandle_t handle) {

  int *csrRowPtrA, *csrColIndA;
  float *csrValA;
  cudaMalloc((void **)&csrRowPtrA, sizeof(int) * (m + 1));
  cudaMalloc((void **)&csrColIndA, sizeof(int) * nnz);
  cudaMalloc((void **)&csrValA, sizeof(float) * nnz);
  cudaMemcpy(csrValA, csrVal, nnz * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrA, csrRowptr, (m + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
  int base, nnzb;
  int mb = (m + blockDim - 1) / blockDim;
  cudaMalloc(bsrRowPtr, sizeof(int) * (mb + 1));
  // nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnzb;
  cusparseXcsr2bsrNnz(handle, dir, m, n, descr, csrRowPtrA, csrColIndA,
                      blockDim, descr, *bsrRowPtr, nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
    nnzb = *nnzTotalDevHostPtr;
  } else {
    cudaMemcpy(&nnzb, *bsrRowPtr + mb, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&base, *bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost);
    nnzb -= base;
  }
  cudaMalloc(bsrColInd, sizeof(int) * nnzb);
  cudaMalloc(bsrVal, sizeof(float) * (blockDim * blockDim) * nnzb);
  cusparseScsr2bsr(handle, dir, m, n, descr, csrValA, csrRowPtrA, csrColIndA,
                   blockDim, descr, *bsrVal, *bsrRowPtr, *bsrColInd);
  *nnzblocks = nnzb;
  *n_block_rows = mb;
  cudaFree(csrRowPtrA);
  cudaFree(csrColIndA);
  cudaFree(csrValA);
}
void csr2bsr(int blockDim, int n, int m, int nnz, int *csrRowptr,
             int *csrColInd, double *csrVal, int **bsrRowPtr, int **bsrColInd,
             double **bsrVal, int *nnzblocks, int *n_block_rows,
             cusparseHandle_t handle) {

  int *csrRowPtrA, *csrColIndA;
  double *csrValA;
  cudaMalloc((void **)&csrRowPtrA, sizeof(int) * (m + 1));
  cudaMalloc((void **)&csrColIndA, sizeof(int) * nnz);
  cudaMalloc((void **)&csrValA, sizeof(double) * nnz);
  cudaMemcpy(csrValA, csrVal, nnz * sizeof(double), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrA, csrRowptr, (m + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDirection_t dir = CUSPARSE_DIRECTION_COLUMN;
  int base, nnzb;
  int mb = (m + blockDim - 1) / blockDim;
  cudaMalloc(bsrRowPtr, sizeof(int) * (mb + 1));
  // nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnzb;
  cusparseXcsr2bsrNnz(handle, dir, m, n, descr, csrRowPtrA, csrColIndA,
                      blockDim, descr, *bsrRowPtr, nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
    nnzb = *nnzTotalDevHostPtr;
  } else {
    cudaMemcpy(&nnzb, *bsrRowPtr + mb, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&base, *bsrRowPtr, sizeof(int), cudaMemcpyDeviceToHost);
    nnzb -= base;
  }
  cudaMalloc(bsrColInd, sizeof(int) * nnzb);
  cudaMalloc(bsrVal, sizeof(double) * (blockDim * blockDim) * nnzb);
  cusparseDcsr2bsr(handle, dir, m, n, descr, csrValA, csrRowPtrA, csrColIndA,
                   blockDim, descr, *bsrVal, *bsrRowPtr, *bsrColInd);
  *nnzblocks = nnzb;
  *n_block_rows = mb;
  cudaFree(csrRowPtrA);
  cudaFree(csrColIndA);
  cudaFree(csrValA);
}
#include "matrix_converter.h"

/*
template <typename data_type>
__global__ void csr2ell(volatile unsigned int *__restrict__ data,
                        volatile data_type *__restrict__ columns,
                        const int *const row_ptr, const int *const col,
                        const data_type *const val, const int n,
                        const int elements_per_ell) {
  register int start;
  register int end;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    start = row_ptr[TID];
    end = row_ptr[TID + 1];
    for (register int element = start;
         element < min(start + elements_per_ell, end); element++) {
      data[TID + (element - start) * n] = val[element];
      columns[TID + (element - start) * n] = col[element];
    }
  }
}
template <typename data_type>
__global__ void coorowsize(volatile int *__restrict__ cooElementsRow,
                           const int *const row_ptr, const int *const col,
                           const data_type *const val, const int n,
                           const int elements_per_ell) {
  register int start;
  register int end;
  register int row_size for (register int TID =
                                 threadIdx.x + blockIdx.x * blockDim.x;
                             TID < n; TID += gridDim.x * blockDim.x) {
    start = row_ptr[TID];
    end = row_ptr[TID + 1];
    row_size = end - start;
    if (row_size > elements_per_ell) {
      cooElementsRow[TID] = row_size - elements_per_ell;
    } else {
      cooElementsRow[TID] = 0;
    }
  }
}
template <typename data_type>
__global__ void Csr2Coo(data_type *coo_data, int *coo_col_ids, int *coo_row_ids,
                        const int elements_per_ell, int *row_ptr, int *col,
                        data_type *val, const int n) {
  register int start;
  register int end;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    start = row_ptr[TID];
    end = row_ptr[TID + 1];

    for (register int element = elements_per_ell; element < end - start;
         element++) {
    }
  }
}

template <typename data_type> void PrepareHybrid(sparse_matrix<data_type> *P) {
  int elements_per_ell = P->nnz / P->n;
  CUDA_CALL(
      cudaMalloc(&P->ell_data, P->n * elements_per_ell * sizeof(data_type)));
  CUDA_CALL(
      cudaMalloc(&P->ell_cols, P->n * elements_per_ell * sizeof(unsigned int)));
  initKernel<<<64, 1024>>>(P->ell_data, (data_type)0, P->n * elements_per_ell);
  initKernel<<<64, 1024>>>(P->ell_cols, (int)0, P->n * elements_per_ell);
  cudaDeviceSynchronize();
  csr2ell(P->ell_data, P->ell_cols, P->row, P->col, P->val, P->n,
          elements_per_ell);
  thrust::device_vector<int> cooElementsRow(n);
  coorowsize(thrust::raw_pointer_cast(cooElementsRow.data()), P->row, P->col,
             P->val, P->n, elements_per_ell);
  cudaDeviceSynchronize();
  int coo_size =
      thrust::transform_reduce(cooElementsRow.begin(), cooElementsRow.end());

  CUDA_CALL(cudaMalloc(&P->coo_data, coo_size * sizeof(data_type)));
  CUDA_CALL(cudaMalloc(&P->coo_col_ids, coo_size * sizeof(unsigned int)));
  CUDA_CALL(cudaMalloc(&P->coo_row_ids, coo_size * sizeof(unsigned int)));
  Csr2Coo(P->coo_data, P->coo_col_ids, P->coo_row_ids, elements_per_ell, P->row,
          P->col, P->val, n);
  P->coo_size = coo_size;
  P->elements_in_rows = elements_per_ell;
}
*/
template <typename data_type>
void PrepareHybrid(sparse_matrix<data_type> *P) {
  int nnz, rows;
  nnz = P->nnz;
  rows = P->n;
  csr_matrix_class<data_type> A;
  A.nnz = nnz;
  A.n = rows;
  A.data.reset(new data_type[nnz]);
  A.columns.reset(new unsigned int[nnz]);
  A.row_ptr.reset(new unsigned int[rows + 1]);
  cudaMemcpy(A.data.get(), P->val, nnz * sizeof(data_type),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(A.row_ptr.get(), P->row, (rows + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);
  cudaMemcpy(A.columns.get(), P->col, nnz * sizeof(int),
             cudaMemcpyDeviceToHost);
  hybrid_matrix_class<data_type> D(A);
  D.allocate(A, 0.001);
  const size_t A_size = D.ell_matrix->get_matrix_size();
  const size_t col_ids_size = A_size;
  CUDA_CALL(cudaMalloc(&P->ell_data, A_size * sizeof(data_type)));
  CUDA_CALL(cudaMalloc(&P->ell_cols, A_size * sizeof(unsigned int)));
  cudaMemcpy(P->ell_data, D.ell_matrix->data.get(), A_size * sizeof(data_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(P->ell_cols, D.ell_matrix->columns.get(),
             col_ids_size * sizeof(unsigned int), cudaMemcpyHostToDevice);

  const size_t coo_size = D.coo_matrix->get_matrix_size();
  CUDA_CALL(cudaMalloc(&P->coo_data, coo_size * sizeof(data_type)));
  CUDA_CALL(cudaMalloc(&P->coo_col_ids, coo_size * sizeof(unsigned int)));
  CUDA_CALL(cudaMalloc(&P->coo_row_ids, coo_size * sizeof(unsigned int)));

  cudaMemcpy(P->coo_data, D.coo_matrix->data.get(),
             coo_size * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(P->coo_col_ids, D.coo_matrix->cols.get(),
             coo_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(P->coo_row_ids, D.coo_matrix->rows.get(),
             coo_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
  P->coo_size = coo_size;
  P->elements_in_rows = D.ell_matrix->elements_in_rows;

}
template <typename data_type>
sparse_matrix<data_type> *PrepareHybrid(int nnz, int rows, int *csrCidx,
                                        int *csrRptr, data_type *csrVal) {
  sparse_matrix<data_type> *Pd =
      (sparse_matrix<data_type> *)malloc(sizeof(sparse_matrix<data_type>));
  csr_matrix_class<data_type> A;
  A.nnz = nnz;
  A.n = rows;
  A.data.reset(new data_type[nnz]);
  A.columns.reset(new unsigned int[nnz]);
  A.row_ptr.reset(new unsigned int[rows + 1]);
  for (int i = 0; i < nnz; i++) {
    A.columns[i] = (unsigned int)csrCidx[i];
    A.data[i] = csrVal[i];
  }
  for (int i = 0; i < rows + 1; i++) {
    A.row_ptr[i] = csrRptr[i];
  }
  hybrid_matrix_class<data_type> D(A);
  D.allocate(A, 0.001);
  const size_t A_size = D.ell_matrix->get_matrix_size();
  const size_t col_ids_size = A_size;
  CUDA_CALL(cudaMalloc(&Pd->ell_data, A_size * sizeof(data_type)));
  CUDA_CALL(cudaMalloc(&Pd->ell_cols, A_size * sizeof(unsigned int)));
  cudaMemcpy(Pd->ell_data, D.ell_matrix->data.get(), A_size * sizeof(data_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(Pd->ell_cols, D.ell_matrix->columns.get(),
             col_ids_size * sizeof(unsigned int), cudaMemcpyHostToDevice);

  const size_t coo_size = D.coo_matrix->get_matrix_size();
  CUDA_CALL(cudaMalloc(&Pd->coo_data, coo_size * sizeof(data_type)));
  CUDA_CALL(cudaMalloc(&Pd->coo_col_ids, coo_size * sizeof(unsigned int)));
  CUDA_CALL(cudaMalloc(&Pd->coo_row_ids, coo_size * sizeof(unsigned int)));

  cudaMemcpy(Pd->coo_data, D.coo_matrix->data.get(),
             coo_size * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(Pd->coo_col_ids, D.coo_matrix->cols.get(),
             coo_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
  cudaMemcpy(Pd->coo_row_ids, D.coo_matrix->rows.get(),
             coo_size * sizeof(unsigned int), cudaMemcpyHostToDevice);
  Pd->n = rows;
  Pd->nnz = nnz;
  Pd->coo_size = coo_size;
  Pd->elements_in_rows = D.ell_matrix->elements_in_rows;

  return Pd;
}
template <typename data_type>
sparse_matrix<data_type> *PrepareSparseMatrix(sparse_matrix<data_type> *P,
                                              int *perm, int format,
                                              const char *method, int bs) {
  sparse_matrix<data_type> *Pd =
      (sparse_matrix<data_type> *)malloc(sizeof(sparse_matrix<data_type>));
  int N = P->n;
  int M = P->m;
  int n = P->n;
  int nnz = P->nnz;
  int mb = 0;
  int nnzb = 0;
  for (int i = 0; i < n; i++) {
    perm[i] = i;
  }
  if (format == 0) {
    CUDA_CALL(cudaMallocManaged(&Pd->col, nnz * sizeof(matidx)));
    CUDA_CALL(cudaMallocManaged(&Pd->val, nnz * sizeof(data_type)));
    CUDA_CALL(cudaMallocManaged(&Pd->row, (n + 1) * sizeof(matidx)));

    cudaMemcpy(Pd->col, P->row, nnz * sizeof(matidx), cudaMemcpyHostToDevice);
    cudaMemcpy(Pd->val, P->val, nnz * sizeof(data_type),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Pd->row, P->col, (n + 1) * sizeof(matidx),
               cudaMemcpyHostToDevice);
  } else if (format == 2) {
    // csr to coo
    CUDA_CALL(cudaMallocManaged(&Pd->col, nnz * sizeof(matidx)));
    CUDA_CALL(cudaMallocManaged(&Pd->val, nnz * sizeof(data_type)));
    CUDA_CALL(cudaMallocManaged(&Pd->row, nnz * sizeof(matidx)));
    matidx *coorow = (matidx *)malloc(sizeof(matidx) * nnz);
    for (int i = 0; i < n; i++) {
      for (int j = P->col[i]; j < P->col[i + 1]; j++) {
        coorow[j] = i;
      }
    }
    cudaMemcpy(Pd->col, P->row, nnz * sizeof(matidx), cudaMemcpyHostToDevice);
    cudaMemcpy(Pd->val, P->val, nnz * sizeof(data_type),
               cudaMemcpyHostToDevice);
    cudaMemcpy(Pd->row, coorow, nnz * sizeof(matidx), cudaMemcpyHostToDevice);
    free(coorow);
  } else if (format == 1) {
    data_type *csr_val_permuted =
        (data_type *)calloc(P->nnz, sizeof(data_type));
    int *csr_col_permuted = (int *)calloc(P->nnz, sizeof(int));
    int *csr_row_permuted = (int *)calloc(P->n + 1, sizeof(int));
    cusolverSpHandle_t handle = NULL;
    cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */

    cudaStream_t stream = NULL;
    cusparseMatDescr_t descrA = NULL;
    checkCudaErrors(cusolverSpCreate(&handle));
    checkCudaErrors(cusparseCreate(&cusparseHandle));

    checkCudaErrors(cudaStreamCreate(&stream));
    /* bind stream to cusparse and cusolver*/
    checkCudaErrors(cusolverSpSetStream(handle, stream));
    checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

    /* configure matrix descriptor*/
    checkCudaErrors(cusparseCreateMatDescr(&descrA));
    checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
    int baseA = 0; /* base index in CSR format */
    if (baseA) {
      checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
    } else {
      checkCudaErrors(
          cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
    }
    SparseReorder<data_type>(method, handle, descrA, P->m, P->n, P->nnz, P->col,
                             P->row, P->val, csr_row_permuted, csr_col_permuted,
                             csr_val_permuted, perm);
    // csr2bsr(bs, N, M, nnz,  P->col,  P->row, P->val, &Pd->row, &Pd->col,
    // &Pd->val,&nnzb, &mb, cusparseHandle);
    csr2bsr(bs, N, M, nnz, csr_row_permuted, csr_col_permuted, csr_val_permuted,
            &Pd->row, &Pd->col, &Pd->val, &nnzb, &mb, cusparseHandle);
    free(csr_val_permuted);
    free(csr_col_permuted);
    free(csr_row_permuted);
  } else if (format == 3) {
    Pd = PrepareHybrid<data_type>(P->nnz, P->n, P->row, P->col, P->val);
  }
  Pd->n = n;
  Pd->nnz = nnz;
  Pd->blockSize = bs;
  Pd->blockRows = mb;
  Pd->nnzb = nnzb;
  Pd->format = format;
  return Pd;
}
template void PrepareHybrid(sparse_matrix<float> *P);
template void PrepareHybrid(sparse_matrix<double> *P);

template sparse_matrix<float> *PrepareSparseMatrix(sparse_matrix<float> *P,
                                                   int *perm, int format,
                                                   const char *method, int bs);
template sparse_matrix<double> *PrepareSparseMatrix(sparse_matrix<double> *P,
                                                    int *perm, int format,
                                                    const char *method, int bs);
