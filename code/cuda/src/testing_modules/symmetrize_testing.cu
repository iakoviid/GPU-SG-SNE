#include <stdio.h>
#include <assert.h>
#include <random>
#include "../sparsematrix.cuh"
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

/********/
/* MAIN */
/********/
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
int main(int argc, char **argv)
 {

    // --- Initialize cuSPARSE
    cusparseHandle_t handle;    cusparseSafeCall(cusparseCreate(&handle));

    // --- Initialize matrix descriptors
    cusparseMatDescr_t descrA, descrB, descrC;
    cusparseSafeCall(cusparseCreateMatDescr(&descrA));
    cusparseSafeCall(cusparseCreateMatDescr(&descrB));
    cusparseSafeCall(cusparseCreateMatDescr(&descrC));
    int n=atoi(argv[1]);
    sparse_matrix *Ah = generateRandomCSC(n);
    int nnz1=Ah->nnz;
    const int M = n;                                    // --- Number of rows
    const int N = n;                                    // --- Number of columns



    // --- Device vectors defining the block-sparse matrices
    matval *d_csrValA;       gpuErrchk(cudaMalloc(&d_csrValA, nnz1 * sizeof(matval)));
    matidx *d_csrRowPtrA;      gpuErrchk(cudaMalloc(&d_csrRowPtrA, (M + 1) * sizeof(matidx)));
    matidx *d_csrColIndA;      gpuErrchk(cudaMalloc(&d_csrColIndA, nnz1 * sizeof(matidx)));
    gpuErrchk(cudaMemcpy(d_csrValA, Ah->val, nnz1 * sizeof(matval), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_csrRowPtrA, Ah->col, (M + 1) * sizeof(matidx), cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_csrColIndA, Ah->row, nnz1 * sizeof(matidx), cudaMemcpyHostToDevice));


    sparse_matrix A;
    A.m=M;
    A.n=N;
    A.nnz=nnz1;
    A.row=d_csrRowPtrA;
    A.col=d_csrColIndA;
    A.val=d_csrValA;
    sparse_matrix C;
    symmetrizeMatrixGPU(&A,&C);

    // --- Transforming csr to dense format
    //symmetrizeMatrix(&C);
    matval *d_A;             gpuErrchk(cudaMalloc(&d_A, M * N * sizeof(matval)));
    cusparseSafeCall(cusparseDcsr2dense(handle, M, N, descrA, A.val, A.row, A.col, d_A, M));

    matval *h_A = (matval *)malloc(M * N * sizeof(matval));
    gpuErrchk(cudaMemcpy(h_A, d_A, M * N * sizeof(matval), cudaMemcpyDeviceToHost));

    for (int m = 0; m < M; m++) {
        for (int n = 0; n < N; n++) {
            printf("%lf ", h_A[m + n * M]);
            //if(n>10){break;}
        }
        printf("\n");
      //if(m>10){break;}
    }

    return 0;
}
