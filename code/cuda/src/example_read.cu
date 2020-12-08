
#include "cusolverSp.h"
#include "sparsematrix.hpp"
#include "types.hpp"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#include "helper/helper_cuda.h"
#include "sparse_reorder.cuh"

void csr2bsr(int blockDim, int n, int m, int nnz, int *csrRowptr,
             int *csrColInd, coord *csrVal, int **bsrRowPtr, int **bsrColInd,
             coord **bsrVal, int *nnzblocks, int *n_block_rows,
             cusparseHandle_t handle) {

  int *csrRowPtrA, *csrColIndA;
  coord *csrValA;
  cudaMalloc((void **)&csrRowPtrA, sizeof(int) * (m + 1));
  cudaMalloc((void **)&csrColIndA, sizeof(int) * nnz);
  cudaMalloc((void **)&csrValA, sizeof(coord) * nnz);
  cudaMemcpy(csrValA, csrVal, nnz * sizeof(coord), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrA, csrRowptr, (m + 1) * sizeof(int),
             cudaMemcpyHostToDevice);

  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
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
  cudaMalloc(bsrVal, sizeof(coord) * (blockDim * blockDim) * nnzb);
  cusparseDcsr2bsr(handle, dir, m, n, descr, csrValA, csrRowPtrA, csrColIndA,
                   blockDim, descr, *bsrVal, *bsrRowPtr, *bsrColInd);
  *nnzblocks = nnzb;
  *n_block_rows = mb;
  cudaFree(csrRowPtrA);
  cudaFree(csrColIndA);
  cudaFree(csrValA);
}
#define FLAG_BSDB_PERM
int main(int argc, char *argv[]) {
  // int ret_code;
  // MM_typecode matcode;
  // FILE *f;

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
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  }

  int M, N, nz;
  int *I, *J;
  double *val;
  // ReadMatrix(&M,&N,&nz,&I,&J, &val,argc,argv);
  N = atoi(argv[1]);
  M = atoi(argv[2]);
  nz = atoi(argv[3]);
  int bs = atoi(argv[4]);
  I = (int *)malloc(sizeof(int) * nz);
  J = (int *)malloc(sizeof(int) * nz);
  val = (coord *)malloc(sizeof(coord) * nz);

  for (int i = 0; i < nz; i++) {
    scanf("%d %d %lf\n", &J[i], &I[i], &val[i]);
    I[i]--;
    J[i]--;
  }

  /************************/
  /* now write out matrix */
  /************************/
  double *csr_val = (double *)calloc(nz, sizeof(double));
  int *csr_col = (int *)calloc(nz, sizeof(int));
  int *csr_row = (int *)calloc(M + 1, sizeof(int));

  for (int i = 0; i < nz; i++) {
    csr_val[i] = val[i];
    csr_col[i] = J[i];
    csr_row[I[i] + 1]++;
  }
  for (int i = 0; i < M; i++) {
    csr_row[i + 1] += csr_row[i];
  }
  coord *bsrValC;
  int *bsrRowPtrC, *bsrColIndC;
  int mb, nnzb;
  sparse_matrix P;
  P.n = N;
  P.m = M;
  P.nnz = nz;
  P.val = csr_val;
  P.col = csr_row;
  P.row = csr_col;

  symmetrizeMatrix(&P);
  N = P.n;
  M = P.m;
  nz = P.nnz;
  printf("nnz= %d\n", P.nnz);

  // csr_val=P.val;
  // csr_row=P.col;
  // csr_col=P.row;

  //csr2bsr(bs, N, M, nz, P.col, P.row, P.val, &bsrRowPtrC, &bsrColIndC, &bsrValC,&nnzb, &mb, cusparseHandle);
  //printf("nnzb=%d mb=%d\n", nnzb, mb);
  /*
  sparse_matrix P;
  P.n=N;
  P.m=M;
  P.nnz=nz;
  P.row=csr_row;
  P.col=csr_col;
  P.val=csr_val;

  //symmetrizeMatrix( &P );
  // ~~~~~~~~~~ extracting BSDB permutation
  idx_t *perm = static_cast<idx_t *>( malloc(P.n * sizeof(idx_t)) );
  idx_t *iperm = static_cast<idx_t *>( malloc(P.n * sizeof(idx_t)) );

#ifdef FLAG_BSDB_PERM

  std::cout << "Nested dissection permutation..." << std::flush;
  // idx_t options[METIS_NOPTIONS];
  // METIS_SetDefaultOptions(options);
  // options[METIS_OPTION_NUMBERING] = 0;

  int status = METIS_NodeND( &P.n,
                             reinterpret_cast<idx_t *> (P.row),
                             reinterpret_cast<idx_t *> (P.col),
                             NULL, NULL,
                             perm, iperm );


  permuteMatrix( &P, perm, iperm );


  if( status != METIS_OK ) {
    std::cerr << "METIS error."; exit(1);
  }

  std::cout << "DONE" << std::endl;

#else

  for( int i = 0; i < P.n; i++ ){
    perm[i]  = i;
    iperm[i] = i;
  }

#endif
*/
  int *perm = static_cast<int *>(malloc(N * sizeof(int)));
  double *csr_val_permuted = (double *)calloc(nz, sizeof(double));
  int *csr_col_permuted = (int *)calloc(nz, sizeof(int));
  int *csr_row_permuted = (int *)calloc(M + 1, sizeof(int));

  SparseReorder(argv[5], handle, descrA, M, N, nz, P.col, P.row, P.val,
               csr_row_permuted, csr_col_permuted, csr_val_permuted, perm);
/*
  for (int i = 0; i < N; i++) {
    for (int element = csr_row_permuted[i]; element < csr_row_permuted[i + 1];
         element++) {
      printf("%d %d %lf\n", i, csr_col_permuted[element],
             csr_val_permuted[element]);
    }
  }
  */
  csr2bsr(bs, N, M, nz, csr_row_permuted, csr_col_permuted, csr_val_permuted, &bsrRowPtrC, &bsrColIndC, &bsrValC,&nnzb, &mb, cusparseHandle);
  printf("nnzb=%d mb=%d\n", nnzb, mb);

  /*for(int i=0;i<N;i++){
    printf("%d\n",perm[i] );
  }*/
  if (handle) {
    checkCudaErrors(cusolverSpDestroy(handle));
  }
  if (cusparseHandle) {
    checkCudaErrors(cusparseDestroy(cusparseHandle));
  }
  if (stream) {
    checkCudaErrors(cudaStreamDestroy(stream));
  }
  if (descrA) {
    checkCudaErrors(cusparseDestroyMatDescr(descrA));
  }
  return 0;
}
