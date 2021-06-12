#include "sparse_reorder.cuh"

template <class dataPoint>
void SparseReorder(const char *method, cusolverSpHandle_t handle,
                   cusparseMatDescr_t descrA, int rowsA, int colsA, int nnzA,
                   int *h_csrRowPtrA, int *h_csrColIndA, dataPoint *h_csrValA,
                   int *h_csrRowPtrB, int *h_csrColIndB, dataPoint *h_csrValB,
                   int *h_Q) {
  /* verify if A has symmetric pattern or not */
  int issym = 0;

  checkCudaErrors(cusolverSpXcsrissymHost(handle, rowsA, nnzA, descrA,
                                          h_csrRowPtrA, h_csrRowPtrA + 1,
                                          h_csrColIndA, &issym));
  /*
  if (!issym) {
    printf("Error: A has no symmetric pattern \n");
    exit(EXIT_FAILURE);
  }*/
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

    } else if (0 == strcmp(method, "none")) {
      printf("No reordering is chosen, Q = 0:n-1 \n");
      for (int j = 0; j < rowsA; j++) {
        h_Q[j] = j;
      }

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

template void SparseReorder(const char *method, cusolverSpHandle_t handle,
                            cusparseMatDescr_t descrA, int rowsA, int colsA,
                            int nnzA, int *h_csrRowPtrA, int *h_csrColIndA,
                            float *h_csrValA, int *h_csrRowPtrB,
                            int *h_csrColIndB, float *h_csrValB, int *h_Q);

template void SparseReorder(const char *method, cusolverSpHandle_t handle,
                            cusparseMatDescr_t descrA, int rowsA, int colsA,
                            int nnzA, int *h_csrRowPtrA, int *h_csrColIndA,
                            double *h_csrValA, int *h_csrRowPtrB,
                            int *h_csrColIndB, double *h_csrValB, int *h_Q);
