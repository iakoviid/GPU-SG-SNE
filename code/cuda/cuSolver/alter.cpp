
#include <assert.h>
#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "cusolverSp.h"
#include "cusparse.h"

#include "helper_cuda.h"
#include "helper_cusolver.h"

template <typename T_ELEM>
int loadMMSparseMatrix(char *filename, char elem_type, bool csrFormat, int *m,
                       int *n, int *nnz, T_ELEM **aVal, int **aRowInd,
                       int **aColInd, int extendSymMatrix);

void UsageSP(void) {
  printf("<options>\n");
  printf("-h          : display this help\n");
  printf("-R=<name>   : choose a linear solver\n");
  printf("              chol (cholesky factorization), this is default\n");
  printf("              qr   (QR factorization)\n");
  printf("              lu   (LU factorization)\n");
  printf("-P=<name>    : choose a reordering\n");
  printf("              symrcm (Reverse Cuthill-McKee)\n");
  printf("              symamd (Approximate Minimum Degree)\n");
  printf("              metis  (nested dissection)\n");
  printf("-file=<filename> : filename containing a matrix in MM format\n");
  printf("-device=<device_id> : <device_id> if want to run on specific GPU\n");

  exit(0);
}

void parseCommandLineArguments(int argc, char *argv[], struct testOpts &opts) {
  memset(&opts, 0, sizeof(opts));

  if (checkCmdLineFlag(argc, (const char **)argv, "-h")) {
    UsageSP();
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "R")) {
    char *solverType = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "R", &solverType);

    if (solverType) {
      if ((STRCASECMP(solverType, "chol") != 0) &&
          (STRCASECMP(solverType, "lu") != 0) &&
          (STRCASECMP(solverType, "qr") != 0)) {
        printf("\nIncorrect argument passed to -R option\n");
        UsageSP();
      } else {
        opts.testFunc = solverType;
      }
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "P")) {
    char *reorderType = NULL;
    getCmdLineArgumentString(argc, (const char **)argv, "P", &reorderType);

    if (reorderType) {
      if ((STRCASECMP(reorderType, "symrcm") != 0) &&
          (STRCASECMP(reorderType, "symamd") != 0) &&
          (STRCASECMP(reorderType, "metis") != 0)) {
        printf("\nIncorrect argument passed to -P option\n");
        UsageSP();
      } else {
        opts.reorder = reorderType;
      }
    }
  }

  if (checkCmdLineFlag(argc, (const char **)argv, "file")) {
    char *fileName = 0;
    getCmdLineArgumentString(argc, (const char **)argv, "file", &fileName);

    if (fileName) {
      opts.sparse_mat_filename = fileName;
    } else {
      printf("\nIncorrect filename passed to -file \n ");
      UsageSP();
    }
  }
}
void MetisReorder(const char *method, cusolverSpHandle_t handle,
                  cusparseMatDescr_t descrA, int rowsA, int colsA, int nnzA,
                  int *h_csrRowPtrA, int *h_csrColIndA, double *h_csrValA,
                  int *h_csrRowPtrB, int *h_csrColIndB, double *h_csrValB,
                  int *h_Q) {
  /* verify if A has symmetric pattern or not */
  int issym = 0;

  checkCudaErrors(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
                                          h_csrRowPtrA, h_csrRowPtrA + 1,
                                          h_csrColIndA, &issym));
  if (!issym) {
    printf("Error: A has no symmetric pattern \n");
    exit(EXIT_FAILURE);
  }
  printf("step 2: reorder the matrix A to minimize zero fill-in\n");
  printf("        if the user choose a reordering by -P=symrcm, -P=symamd or "
         "-P=metis\n");
  void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */
  int *h_mapBfromA = NULL; /* <int> nnzA */
  size_t size_perm = 0;
  h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);
  assert(NULL != h_mapBfromA);

  if (NULL != method) {
    if (0 == strcmp(method, "symrcm")) {
      printf("Q = symrcm(A) \n");
      checkCudaErrors(cusolverSpXcsrsymrcmHost(
          handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
    } else if (0 == strcmp(method, "symamd")) {
      printf("Q = symamd(A) \n");
      checkCudaErrors(cusolverSpXcsrsymamdHost(
          handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
    } else if (0 == strcmp(method, "metis")) {
      printf("Q = metis(A) \n");
      checkCudaErrors(cusolverSpXcsrmetisndHost(handle, rowsA, nnzA, descrA,
                                                h_csrRowPtrA, h_csrColIndA,
                                                NULL, /* default setting. */
                                                h_Q));
    } else {
      fprintf(stderr, "Error: %s is unknown reordering\n", method);
      exit(1);
    }
  } else {
    printf("No reordering is chosen, Q = 0:n-1 \n");
    for (int j = 0; j < rowsA; j++) {
      h_Q[j] = j;
    }
  }

  printf("B = A(Q,Q) \n");

  memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
  memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

  checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
      handle, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
      &size_perm));

  if (buffer_cpu) {
    free(buffer_cpu);
  }
  buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
  assert(NULL != buffer_cpu);

  /* h_mapBfromA = Identity */
  for (int j = 0; j < nnzA; j++) {
    h_mapBfromA[j] = j;
  }
  checkCudaErrors(cusolverSpXcsrpermHost(handle, rowsA, colsA, nnzA, descrA,
                                         h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
                                         h_mapBfromA, buffer_cpu));

  /* B = A( mapBfromA ) */
  for (int j = 0; j < nnzA; j++) {
    h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
  }
  if (h_mapBfromA) {
    free(h_mapBfromA);
  }
  if (buffer_cpu) {
    free(buffer_cpu);
  }
}

int main(int argc, char *argv[]) {
  struct testOpts opts;
  cusolverSpHandle_t handle = NULL;
  cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
  cudaStream_t stream = NULL;
  cusparseMatDescr_t descrA = NULL;

  int rowsA = 0; /* number of rows of A */
  int colsA = 0; /* number of columns of A */
  int nnzA = 0;  /* number of nonzeros of A */
  int baseA = 0; /* base index in CSR format */

  /* CSR(A) from I/O */
  int *h_csrRowPtrA = NULL;
  int *h_csrColIndA = NULL;
  double *h_csrValA = NULL;

  int *h_Q = NULL; /* <int> n */
                   /* reorder to reduce zero fill-in */
                   /* Q = symrcm(A) or Q = symamd(A) */
  /* B = Q*A*Q' or B = A(Q,Q) by MATLAB notation */
  int *h_csrRowPtrB = NULL; /* <int> n+1 */
  int *h_csrColIndB = NULL; /* <int> nnzA */
  double *h_csrValB = NULL; /* <double> nnzA */
  int *h_mapBfromA = NULL;  /* <int> nnzA */

  size_t size_perm = 0;
  void *buffer_cpu = NULL; /* working space for permutation: B = Q*A*Q^T */

  double tol = 1.e-12;
  const int reorder = 0; /* no reordering */
  int singularity = 0;   /* -1 if A is invertible under tol. */

  /* the constants are used in residual evaluation, r = b - A*x */
  const double minus_one = -1.0;
  const double one = 1.0;

  double b_inf = 0.0;
  double x_inf = 0.0;
  double r_inf = 0.0;
  double A_inf = 0.0;
  int errors = 0;
  int issym = 0;

  double start, stop;
  double time_solve_cpu;
  double time_solve_gpu;

  parseCommandLineArguments(argc, argv, opts);

  if (NULL == opts.testFunc) {
    opts.testFunc = "chol"; /* By default running Cholesky as NO solver selected
                               with -R option. */
  }

  findCudaDevice(argc, (const char **)argv);

  if (opts.sparse_mat_filename == NULL) {
    opts.sparse_mat_filename = sdkFindFilePath("lap2D_5pt_n100.mtx", argv[0]);
    if (opts.sparse_mat_filename != NULL)
      printf("Using default input file [%s]\n", opts.sparse_mat_filename);
    else
      printf("Could not find lap2D_5pt_n100.mtx\n");
  } else {
    printf("Using input file [%s]\n", opts.sparse_mat_filename);
  }

  printf("step 1: read matrix market format\n");

  if (opts.sparse_mat_filename == NULL) {
    fprintf(stderr, "Error: input matrix is not provided\n");
    return EXIT_FAILURE;
  }

  if (loadMMSparseMatrix<double>(opts.sparse_mat_filename, 'd', true, &rowsA,
                                 &colsA, &nnzA, &h_csrValA, &h_csrRowPtrA,
                                 &h_csrColIndA, true)) {
    exit(EXIT_FAILURE);
  }
  baseA = h_csrRowPtrA[0]; // baseA = {0,1}
  printf("sparse matrix A is %d x %d with %d nonzeros, base=%d\n", rowsA, colsA,
         nnzA, baseA);

  if (rowsA != colsA) {
    fprintf(stderr, "Error: only support square matrix\n");
    return 1;
  }

  checkCudaErrors(cusolverSpCreate(&handle));
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  checkCudaErrors(cudaStreamCreate(&stream));
  /* bind stream to cusparse and cusolver*/
  checkCudaErrors(cusolverSpSetStream(handle, stream));
  checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

  /* configure matrix descriptor*/
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  if (baseA) {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
  } else {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  }

  h_Q = (int *)malloc(sizeof(int) * colsA);
  h_csrRowPtrB = (int *)malloc(sizeof(int) * (rowsA + 1));
  h_csrColIndB = (int *)malloc(sizeof(int) * nnzA);
  h_csrValB = (double *)malloc(sizeof(double) * nnzA);
  h_mapBfromA = (int *)malloc(sizeof(int) * nnzA);

  assert(NULL != h_Q);
  assert(NULL != h_csrRowPtrB);
  assert(NULL != h_csrColIndB);
  assert(NULL != h_csrValB);
  assert(NULL != h_mapBfromA);

  /* verify if A has symmetric pattern or not */
  /*
    checkCudaErrors(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
                                            h_csrRowPtrA, h_csrRowPtrA + 1,
                                            h_csrColIndA, &issym));

    printf("step 2: reorder the matrix A to minimize zero fill-in\n");
    printf("        if the user choose a reordering by -P=symrcm, -P=symamd or "
           "-P=metis\n");

    if (NULL != opts.reorder) {
      if (0 == strcmp(opts.reorder, "symrcm")) {
        printf("step 2.1: Q = symrcm(A) \n");
        checkCudaErrors(cusolverSpXcsrsymrcmHost(
            handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
      } else if (0 == strcmp(opts.reorder, "symamd")) {
        printf("step 2.1: Q = symamd(A) \n");
        checkCudaErrors(cusolverSpXcsrsymamdHost(
            handle, rowsA, nnzA, descrA, h_csrRowPtrA, h_csrColIndA, h_Q));
      } else if (0 == strcmp(opts.reorder, "metis")) {
        printf("step 2.1: Q = metis(A) \n");
        checkCudaErrors(cusolverSpXcsrmetisndHost(handle, rowsA, nnzA, descrA,
                                                  h_csrRowPtrA, h_csrColIndA,
                                                  NULL,
                                                  h_Q));
        printf("h_Q= %d %d %d\n", h_Q[0], h_Q[1], h_Q[2]);
      } else {
        fprintf(stderr, "Error: %s is unknown reordering\n", opts.reorder);
        return 1;
      }
    } else {
      printf("step 2.1: no reordering is chosen, Q = 0:n-1 \n");
      for (int j = 0; j < rowsA; j++) {
        h_Q[j] = j;
      }
    }

    printf("step 2.2: B = A(Q,Q) \n");

    memcpy(h_csrRowPtrB, h_csrRowPtrA, sizeof(int) * (rowsA + 1));
    memcpy(h_csrColIndB, h_csrColIndA, sizeof(int) * nnzA);

    checkCudaErrors(cusolverSpXcsrperm_bufferSizeHost(
        handle, rowsA, colsA, nnzA, descrA, h_csrRowPtrB, h_csrColIndB, h_Q,
    h_Q, &size_perm));

    if (buffer_cpu) {
      free(buffer_cpu);
    }
    buffer_cpu = (void *)malloc(sizeof(char) * size_perm);
    assert(NULL != buffer_cpu);


    for (int j = 0; j < nnzA; j++) {
      h_mapBfromA[j] = j;
    }
    checkCudaErrors(cusolverSpXcsrpermHost(handle, rowsA, colsA, nnzA, descrA,
                                           h_csrRowPtrB, h_csrColIndB, h_Q, h_Q,
                                           h_mapBfromA, buffer_cpu));

    for (int j = 0; j < nnzA; j++) {
      h_csrValB[j] = h_csrValA[h_mapBfromA[j]];
    }
    */
  MetisReorder(opts.reorder, handle, descrA, rowsA, colsA, nnzA, h_csrRowPtrA,
               h_csrColIndA, h_csrValA, h_csrRowPtrB, h_csrColIndB, h_csrValB,
               h_Q);
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

  if (h_csrValA) {
    free(h_csrValA);
  }
  if (h_csrRowPtrA) {
    free(h_csrRowPtrA);
  }
  if (h_csrColIndA) {
    free(h_csrColIndA);
  }

  if (h_Q) {
    free(h_Q);
  }

  if (h_csrRowPtrB) {
    free(h_csrRowPtrB);
  }
  if (h_csrColIndB) {
    free(h_csrColIndB);
  }
  if (h_csrValB) {
    free(h_csrValB);
  }
  if (h_mapBfromA) {
    free(h_mapBfromA);
  }

  if (buffer_cpu) {
    free(buffer_cpu);
  }

  return 0;
}
