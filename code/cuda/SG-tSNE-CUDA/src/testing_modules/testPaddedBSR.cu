#include <stdint.h>
#include "utils_cuda.cuh"
#include "timer.h"
#include <fstream>
#include <iostream>
#include "cusolverSp.h"
#include "sparsematrix.hpp"
#include "types.hpp"
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "pq.cuh"
#include "helper/helper_cuda.h"
#include "sparse_reorder.cuh"
using namespace std;
#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

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

template <class dataPoint>
dataPoint maxerror(dataPoint * dw, dataPoint *dv, int n, int d) {
  dataPoint *w=(dataPoint* )malloc(n*d*sizeof(dataPoint));
  dataPoint *v=(dataPoint* )malloc(n*d*sizeof(dataPoint));
  CUDA_CALL(cudaMemcpy(v, dv, n * d * sizeof(coord),cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(w, dw, n * d * sizeof(coord),cudaMemcpyDeviceToHost));

  dataPoint maxError = 0;
  dataPoint avgError = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if ((v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]) >
          maxError) {
        maxError =
            (v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]);
          }
      avgError += (v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]);
    }
  }
  free(w);
  free(v);
  return maxError;
}

#define FLAG_BSDB_PERM
int main(int argc, char *argv[]) {
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
  struct GpuTimer timer;

  int M, N, nz;
  int *I, *J;
  double *val;
  N = atoi(argv[1]);
  M = atoi(argv[2]);
  nz = atoi(argv[3]);
  int bs = atoi(argv[4]);
  int d=atoi(argv[6]);
  I = (int *)malloc(sizeof(int) * nz);
  J = (int *)malloc(sizeof(int) * nz);
  val = (coord *)malloc(sizeof(coord) * nz);

  for (int i = 0; i < nz; i++) {
    scanf("%d %d %lf\n", &J[i], &I[i], &val[i]);
    I[i]--;
    J[i]--;
  }

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

  int *perm = static_cast<int *>(malloc(N * sizeof(int)));
  double *csr_val_permuted = (double *)calloc(nz, sizeof(double));
  int *csr_col_permuted = (int *)calloc(nz, sizeof(int));
  int *csr_row_permuted = (int *)calloc(M + 1, sizeof(int));

  SparseReorder(argv[5], handle, descrA, M, N, nz, P.col, P.row, P.val,
               csr_row_permuted, csr_col_permuted, csr_val_permuted, perm);
  csr2bsr(bs, N, M, nz, csr_row_permuted, csr_col_permuted, csr_val_permuted, &bsrRowPtrC, &bsrColIndC, &bsrValC,&nnzb, &mb, cusparseHandle);
  int n=N;

  printf("nnzb=%d mb=%d\n", nnzb, mb);
  coord  *x = (coord   *)malloc(n * d * sizeof(coord ));
  coord  *Fattr;
  coord  *y;
  for(int i=0;i<M*d;i++){
    x[i]=100*(double)rand()/RAND_MAX;
  }
  CUDA_CALL(cudaMallocManaged(&Fattr, n * d * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&y, n * d * sizeof(coord)));
  initKernel<<<64, 1024>>>(Fattr, 0.0, n * d);
  CUDA_CALL(cudaMemcpy(y, x, n * d * sizeof(coord),cudaMemcpyHostToDevice));

  for(int i=0; i<10;i++){
  timer.Start();
  AttractiveEstimation(bsrRowPtrC, bsrColIndC, bsrValC,Fattr, y, n, d, bs,mb,nnzb,nz,1);
  timer.Stop();
  printf("time bsr %f \n",timer.Elapsed() );
  }

  coord  *Fattr2;
  CUDA_CALL(cudaMallocManaged(&Fattr2, n * d * sizeof(coord)));
  initKernel<<<64, 1024>>>(Fattr2, 0.0, n * d);
  int*csrRow,*csrCol;
  coord *csrVal;
  CUDA_CALL(cudaMallocManaged(&csrVal, nz * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&csrCol,nz * sizeof(int)));
  CUDA_CALL(cudaMallocManaged(&csrRow,(M + 1)* sizeof(int)));
  CUDA_CALL(cudaMemcpy(csrVal, csr_val_permuted, nz * sizeof(coord),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(csrRow, csr_row_permuted, (M+1) * sizeof(int),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(csrCol, csr_col_permuted, nz * sizeof(int),cudaMemcpyHostToDevice));

  for(int i=0; i<10;i++){
    timer.Start();
  AttractiveEstimation(csrRow, csrCol, csrVal,Fattr2, y, n, d, bs,mb,nnzb,nz,0);
  timer.Stop();
  printf("time csr %f \n",timer.Elapsed() );
}

  printf("maxError %lf\n",maxerror(Fattr,Fattr2,n,d) );
  free(csr_val_permuted);
  free(csr_row_permuted);
  free(csr_col_permuted);

  /*
  int *bsrRow=(int *)malloc(sizeof(int)*(mb+1));
  int *bsrCol=(int *)malloc(sizeof(int)*(nnzb));
  coord *bsrVal=(coord *)malloc(sizeof(coord)*(nnzb*bs*bs));
  CUDA_CALL(cudaMemcpy(bsrVal, bsrValC, nnzb*bs*bs * sizeof(coord),cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(bsrRow, bsrRowPtrC, (mb+1) * sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(bsrCol, bsrColIndC, (nnzb)* sizeof(int),cudaMemcpyDeviceToHost));
  ofstream myfile;

  myfile.open("bsr.txt");
  for(int i=0;i<mb+1;i++){
    myfile <<bsrRow[i] <<"\n";
  }
  for(int i=0;i<nnzb*bs*bs;i++){
    myfile <<bsrVal[i] <<"\n";
  }
  for(int i=0;i<nnzb;i++){
    myfile <<bsrCol[i] <<"\n";
  }
  myfile.close();
*/
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
