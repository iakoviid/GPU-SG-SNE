
#include <stdio.h>
#include <cusparse.h>
#include "../types.hpp"
#include "../sparsematrix.hpp"
#include <random>

sparse_matrix *generateRandomCSC(int n){

  sparse_matrix *P = (sparse_matrix *) malloc(sizeof(sparse_matrix));

  P->n = n; P->m = n;

  P->col = (matidx *) malloc( (n+1)*sizeof(matidx) );

  for (int j=0 ; j<n ; j++)
    P->col[j] = rand() % 10 + 2;

  int cumsum = 0;
  for(int i = 0; i < P->n; i++){
    int temp = P->col[i];
    P->col[i] = cumsum;
    cumsum += temp;
  }
  P->col[P->n] = cumsum;
  P->nnz = cumsum;

  P->row = (matidx *) malloc( (P->nnz)*sizeof(matidx) );
  P->val = (matval *) malloc( (P->nnz)*sizeof(matval) );

  std::uniform_real_distribution<double> unif(0,1);
  std::default_random_engine re;

  for (int l = 0; l < P->nnz; l++){
    P->row[l] = rand() % n;
    P->val[l] = unif(re);
  }

  return P;

}
/*
void symmetrizeMatrixGPU(sparse_matrix *A, sparse_matrix *C,cusparseHandle_t &handle) {
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

  // Do the transpose operation
  cusparseDcsr2csc(handle, A->m, A->n, A->nnz, A->val, A->row, A->col,
                   csc_values_at, csc_row_ptr_at, csc_column_ptr_at,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);

  cudaDeviceSynchronize();

  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
  // --- Summing the two matrices
  int baseC, nnz3;
  // --- nnzTotalDevHostPtr points to host memory
  int32_t symmetrized_num_nonzeros = -1;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cudaMalloc(&C->row, (M + 1) * sizeof(int));

  cusparseXcsrgeamNnz(handle, M, N, descrA, A->nnz, A->row, A->col, descrA, nnz,
                      csc_column_ptr_at, csc_row_ptr_at, descrA, C->row,
                      &symmetrized_num_nonzeros);
  if (-1 != symmetrized_num_nonzeros) {
    nnz3 = symmetrized_num_nonzeros;
  } else {
    cudaMemcpy(&nnz3, C->row + M, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, C->row, sizeof(int), cudaMemcpyDeviceToHost);
    nnz3 -= baseC;
  }
  cudaMalloc(&(C->col), nnz3 * sizeof(int));
  cudaMalloc(&(C->val), nnz3 * sizeof(matval));
  matval alpha = (matval)1 / 2, beta = (matval)1 / 2;
  cusparseDcsrgeam(handle, M, N, &alpha, descrA, nnz, A->val, A->row, A->col,
                   &beta, descrA, nnz, csc_values_at, csc_column_ptr_at,
                   csc_row_ptr_at, descrA, C->val, C->row, C->col);

  C->nnz = nnz3;
  C->m = M;
  C->n = N;
  cudaDeviceSynchronize();

  cudaFree(csc_column_ptr_at);
  cudaFree(csc_row_ptr_at);
  cudaFree(csc_values_at);
}*/
void symmetrizeMatrixGPU(sparse_matrix *A, sparse_matrix *C,cusparseHandle_t handle) {

  double *d_csrValB;       cudaMalloc(&d_csrValB, A->nnz * sizeof(double));
  int *d_csrRowPtrB;      cudaMalloc(&d_csrRowPtrB, (A->m + 1) * sizeof(int));
  int *d_csrColIndB;      cudaMalloc(&d_csrColIndB, A->nnz * sizeof(int));

  cusparseDcsr2csc(handle, A->m, A->n, A->nnz, A->val, A->row,A->col,
                   d_csrValB, d_csrColIndB, d_csrRowPtrB,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  cudaDeviceSynchronize();
  // --- Summing the two matrices
  int baseC, nnz3;
  cusparseMatDescr_t descrA, descrB, descrC;
  cusparseCreateMatDescr(&descrA);
  cusparseCreateMatDescr(&descrB);
  cusparseCreateMatDescr(&descrC);
  // --- nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnz3;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  cudaMalloc(&C->row, (A->m + 1) * sizeof(int));
  cusparseXcsrgeamNnz(handle, A->m, A->n, descrA, A->nnz, A->row, A->col, descrB, A->nnz, d_csrRowPtrB, d_csrColIndB, descrC, C->row, nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
      nnz3 = *nnzTotalDevHostPtr;
  }
  else{
      cudaMemcpy(&nnz3, C->row + A->m, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, C->row, sizeof(int), cudaMemcpyDeviceToHost);
      nnz3 -= baseC;
  }

  cudaMalloc(&C->col, nnz3 * sizeof(int));
  cudaMalloc(&C->val, nnz3 * sizeof(double));
  double alpha = 0.5, beta = 0.5;
  cusparseDcsrgeam(handle, A->m, A->n, &alpha, descrA, A->nnz, A->val, A->row, A->col, &beta, descrB, A->nnz, d_csrValB, d_csrRowPtrB, d_csrColIndB, descrC, C->val, C->row, C->col);
  cudaDeviceSynchronize();
  C->nnz=nnz3;
  C->m=A->m;
  C->n=A->n;
  cudaFree(d_csrValB);
  cudaFree(d_csrRowPtrB);
  cudaFree(d_csrColIndB);

}
int main(int argc, char **argv)
 {
   int n=atoi(argv[1]);
   cusparseHandle_t handle;    cusparseCreate(&handle);
   sparse_matrix *Ah = generateRandomCSC(n);
   int nnz=Ah->nnz;

   printf("Ah->nnz=%d  \n",Ah->nnz);

   const int M = n;                                    // --- Number of rows
   const int N = n;                                    // --- Number of columns
   sparse_matrix A;
   A.m=M;
   A.n=N;
   A.nnz=nnz;
   cudaMalloc(&A.val, nnz * sizeof(matval));
   cudaMalloc(&A.row, (M + 1) * sizeof(matidx));
   cudaMalloc(&A.col, nnz * sizeof(matidx));
   cudaMemcpy(A.val, Ah->val, nnz * sizeof(matval), cudaMemcpyHostToDevice);
   cudaMemcpy(A.row, Ah->col, (M + 1) * sizeof(matidx), cudaMemcpyHostToDevice);
   cudaMemcpy(A.col, Ah->row, nnz * sizeof(matidx), cudaMemcpyHostToDevice);
   sparse_matrix C;

   symmetrizeMatrixGPU(&A,&C,handle);
   double* val=(double*)malloc(C.nnz*sizeof(double));
   int* row=(int*)malloc(C.nnz*sizeof(int));
   int* col=(int*)malloc(C.nnz*sizeof(int));
   cudaMemcpy(val, C.val, C.nnz * sizeof(matval), cudaMemcpyDeviceToHost);
   cudaMemcpy(row, C.row, (C.n + 1) * sizeof(matidx), cudaMemcpyDeviceToHost);
   cudaMemcpy(col, C.col, C.nnz * sizeof(matidx), cudaMemcpyDeviceToHost);
   symmetrizeMatrix(Ah);
   printf("Ah->nnz=%d vs C.nnz=%d \n",Ah->nnz,C.nnz );
   for(int i=0;i<C.nnz;i++){
     printf("%lf vs %lf\n",Ah->val[i],val[i] );
   }

       return 0;
   }
