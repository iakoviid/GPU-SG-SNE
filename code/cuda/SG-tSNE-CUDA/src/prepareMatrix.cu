#include "prepareMatrix.cuh"
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
#include "matrix_converter.h"

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
  gpuErrchk(cudaMalloc(&P->ell_data, A_size * sizeof(data_type)));
  gpuErrchk(cudaMalloc(&P->ell_cols, A_size * sizeof(unsigned int)));
  cudaMemcpy(P->ell_data, D.ell_matrix->data.get(), A_size * sizeof(data_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(P->ell_cols, D.ell_matrix->columns.get(),
             col_ids_size * sizeof(unsigned int), cudaMemcpyHostToDevice);

  const size_t coo_size = D.coo_matrix->get_matrix_size();
  gpuErrchk(cudaMalloc(&P->coo_data, coo_size * sizeof(data_type)));
  gpuErrchk(cudaMalloc(&P->coo_col_ids, coo_size * sizeof(unsigned int)));
  gpuErrchk(cudaMalloc(&P->coo_row_ids, coo_size * sizeof(unsigned int)));

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
  gpuErrchk(cudaMalloc(&Pd->ell_data, A_size * sizeof(data_type)));
  gpuErrchk(cudaMalloc(&Pd->ell_cols, A_size * sizeof(unsigned int)));
  cudaMemcpy(Pd->ell_data, D.ell_matrix->data.get(), A_size * sizeof(data_type),
             cudaMemcpyHostToDevice);
  cudaMemcpy(Pd->ell_cols, D.ell_matrix->columns.get(),
             col_ids_size * sizeof(unsigned int), cudaMemcpyHostToDevice);

  const size_t coo_size = D.coo_matrix->get_matrix_size();
  gpuErrchk(cudaMalloc(&Pd->coo_data, coo_size * sizeof(data_type)));
  gpuErrchk(cudaMalloc(&Pd->coo_col_ids, coo_size * sizeof(unsigned int)));
  gpuErrchk(cudaMalloc(&Pd->coo_row_ids, coo_size * sizeof(unsigned int)));

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
template void PrepareHybrid(sparse_matrix<float> *P);
template void PrepareHybrid(sparse_matrix<double> *P);
