#include <cusparse.h>
#include <sys/time.h>

void csr2bsr(int blockDim, int n, int m, int nnz, int *csrRowptr,
             int *csrColInd, coord *csrVal, int **bsrRowPtr, int **bsrColInd,
             coord **bsrVal, int* nnzblocks,int* n_block_rows) {

  int *csrRowPtrA, *csrColIndA;
  coord *csrValA;
  cudaMalloc((void **)&csrRowPtrA, sizeof(int) * (m + 1));
  cudaMalloc((void **)&csrColIndA, sizeof(int) * nnz);
  cudaMalloc((void **)&csrValA, sizeof(coord) * nnz);
  cudaMemcpy(csrValA, csrVal, nnz * sizeof(coord), cudaMemcpyHostToDevice);
  cudaMemcpy(csrColIndA, csrColInd, nnz * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(csrRowPtrA, csrRowptr, (m + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cusparseHandle_t handle;
  cusparseCreate(&handle);
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
  *nnzblocks=nnzb;
  *n_block_rows=mb;
  cudaFree(csrRowPtrA);
  cudaFree(csrColIndA);
  cudaFree(csrValA);
}
int main(int argc, char **argv){

return 0;
}
