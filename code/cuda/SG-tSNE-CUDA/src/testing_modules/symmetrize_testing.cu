#include <stdio.h>
#include <assert.h>
#include "../types.hpp"
#include "../sparsematrix.hpp"
#include <cusparse.h>

/*******************/
/* iDivUp FUNCTION */
/*******************/
int iDivUp(int a, int b){ return ((a % b) != 0) ? (a / b + 1) : (a / b); }

/********************/
/* CUDA ERROR CHECK */
/********************/
// --- Credit to http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true)
{
    if (code != cudaSuccess)
    {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) { exit(code); }
    }
}

void gpuErrchk(cudaError_t ans) { gpuAssert((ans), __FILE__, __LINE__); }

/***************************/
/* CUSPARSE ERROR CHECKING */
/***************************/
static const char *_cusparseGetErrorEnum(cusparseStatus_t error)
{
    switch (error)
    {

    case CUSPARSE_STATUS_SUCCESS:
        return "CUSPARSE_STATUS_SUCCESS";

    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "CUSPARSE_STATUS_NOT_INITIALIZED";

    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "CUSPARSE_STATUS_ALLOC_FAILED";

    case CUSPARSE_STATUS_INVALID_VALUE:
        return "CUSPARSE_STATUS_INVALID_VALUE";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "CUSPARSE_STATUS_ARCH_MISMATCH";

    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "CUSPARSE_STATUS_MAPPING_ERROR";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "CUSPARSE_STATUS_EXECUTION_FAILED";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "CUSPARSE_STATUS_INTERNAL_ERROR";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED";

    case CUSPARSE_STATUS_ZERO_PIVOT:
        return "CUSPARSE_STATUS_ZERO_PIVOT";
    }

    return "<unknown>";
}

inline void __cusparseSafeCall(cusparseStatus_t err, const char *file, const int line)
{
    if (CUSPARSE_STATUS_SUCCESS != err) {
        fprintf(stderr, "CUSPARSE error in file '%s', line %d, error %s\nterminating!\n", __FILE__, __LINE__, \
            _cusparseGetErrorEnum(err)); \
            assert(0); \
    }
}

extern "C" void cusparseSafeCall(cusparseStatus_t err) { __cusparseSafeCall(err, __FILE__, __LINE__); }


/* (P+P^T)/2*/
void symmetrizeMatrixGPU(sparse_matrix<matval> *A,cusparseHandle_t &handle) {
  coord *d_csrValB;       gpuErrchk(cudaMalloc(&d_csrValB, A->nnz * sizeof(coord)));
  int *d_csrRowPtrB;      gpuErrchk(cudaMalloc(&d_csrRowPtrB, (A->m + 1) * sizeof(int)));
  int *d_csrColIndB;      gpuErrchk(cudaMalloc(&d_csrColIndB, A->nnz * sizeof(int)));

  cusparseScsr2csc(handle, A->m, A->n, A->nnz, A->val, A->row,A->col,
                   d_csrValB, d_csrColIndB, d_csrRowPtrB,
                   CUSPARSE_ACTION_NUMERIC, CUSPARSE_INDEX_BASE_ZERO);
  cudaDeviceSynchronize();
  // --- Summing the two matrices
  int baseC, nnz3;
  cusparseMatDescr_t descrA;
  cusparseCreateMatDescr(&descrA);
  coord* sym_val;
  int * sym_col,* sym_row;
  // --- nnzTotalDevHostPtr points to host memory
  int *nnzTotalDevHostPtr = &nnz3;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  gpuErrchk(cudaMalloc(&sym_row, (A->m + 1) * sizeof(int)));
  cusparseXcsrgeamNnz(handle, A->m, A->n, descrA, A->nnz, A->row, A->col, descrA, A->nnz, d_csrRowPtrB, d_csrColIndB, descrA, sym_row, nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
      nnz3 = *nnzTotalDevHostPtr;
  }
  else{
      gpuErrchk(cudaMemcpy(&nnz3, sym_row + A->m, sizeof(int), cudaMemcpyDeviceToHost));
      gpuErrchk(cudaMemcpy(&baseC, sym_row, sizeof(int), cudaMemcpyDeviceToHost));
      nnz3 -= baseC;
  }

  gpuErrchk(cudaMalloc(&sym_col, nnz3 * sizeof(int)));
  gpuErrchk(cudaMalloc(&sym_val, nnz3 * sizeof(coord)));
  coord alpha = 0.5, beta = 0.5;
  cusparseScsrgeam(handle, A->m, A->n, &alpha, descrA, A->nnz, A->val, A->row, A->col, &beta, descrA, A->nnz, d_csrValB, d_csrRowPtrB, d_csrColIndB, descrA, sym_val, sym_row, sym_col);
  cudaDeviceSynchronize();
  A->nnz=nnz3;

  gpuErrchk(cudaFree(d_csrValB));
  gpuErrchk(cudaFree(d_csrRowPtrB));
  gpuErrchk(cudaFree(d_csrColIndB));
  gpuErrchk(cudaFree(A->row)); A->row = sym_row;
  gpuErrchk(cudaFree(A->col)); A->col = sym_col;
  gpuErrchk(cudaFree(A->val)); A->val = sym_val;
}
int main() {

    // --- Initialize cuSPARSE
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

    // --- Initialize matrix descriptors
    cusparseMatDescr_t descrA, descrB, descrC;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseCreateMatDescr(&descrB));
    cusparseSafeCall(cusparseCreateMatDescr(&descrC));

    const int M = 6;                                    // --- Number of rows
    const int N = 6;                                    // --- Number of columns

    const int nnz1 = 5400000;                                // --- Number of non-zero blocks for matrix A

    // --- Host vectors defining the first block-sparse matrix
    sparse_matrix<coord> Ah;
    Ah.val = (coord *)malloc(nnz1 * sizeof(coord));
    Ah.row = (int *)malloc((M + 1) * sizeof(int));
    Ah.col = (int *)malloc(nnz1 * sizeof(int));
    for(int i=0;i<n;<nnz1){
      Ah.val[i]
    }



    // --- Device vectors defining the block-sparse matrices
    sparse_matrix<coord> A;
       gpuErrchk(cudaMalloc(&(A.val), nnz1 * sizeof(coord)));
        gpuErrchk(cudaMalloc(&(A.row), (M + 1) * sizeof(int)));
        gpuErrchk(cudaMalloc(&(A.col), nnz1 * sizeof(int)));


    gpuErrchk(cudaMemcpy(A.val, Ah.val, nnz1 * sizeof(coord), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(A.row, Ah.row, (M + 1) * sizeof(int), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(A.col, Ah.col, nnz1 * sizeof(int), cudaMemcpyHostToDevice));
    A.n=N;
    A.nnz=nnz1;
    A.m=M;
    symmetrizeMatrixGPU(&A,handle);


    // --- Transforming csr to dense format
    coord *d_C;             gpuErrchk(cudaMalloc(&d_C, M * N * sizeof(coord)));
    cusparseSafeCall(cusparseScsr2dense(handle, M, N, descrC, A.val, A.row, A.col, d_C, M));

    coord *h_C = (coord *)malloc(M * N * sizeof(coord));
    gpuErrchk(cudaMemcpy(h_C, d_C, M * N * sizeof(coord), cudaMemcpyDeviceToHost));

    // --- m is row index, n column index
    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            printf("%lf ", h_C[m + n * M]);
        }
        printf("\n");
    }

    return 0;
}
