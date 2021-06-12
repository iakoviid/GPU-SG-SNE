/*!
  \file   sgtsne.cpp
  \brief  Entry point to SG-t-SNE

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.


*/
#include "sgtsne.cuh"
#define CUDART_INF_F __uint2double_rn(0x7ff0000000000000)
#define CUDART_NINF_F __uint2double_rn(0xfff0000000000000)
#include "graph_rescaling.hpp"
#include "prepareMatrix.cu"
template <class dataPoint>
dataPoint *generateRandomCoord(int n, int d) {

  dataPoint *y = (dataPoint *)malloc(n * d * sizeof(dataPoint));

  for (int i = 0; i < n * d; i++)
    y[i] = ((dataPoint)rand() / (RAND_MAX)) * .0001;

  return y;
}
template <class dataPoint>
dataPoint *sgtsneCUDA(sparse_matrix* P, tsneparams params, dataPoint *y_in,
                  double **timeInfo) {
  // ~~~~~~~~~~ unless h is specified, use default ones
  if (params.h <= 0)
    switch (params.d) {
    case 1:
      params.h = 0.5;
      break;
    case 2:
      params.h = 0.7;
      break;
    case 3:
      params.h = 1.2;
      break;
    }

  // ~~~~~~~~~~ print input parameters
  printParams(params);

  // ~~~~~~~~~~ make sure input matrix is column stochastic
  uint32_t nStoch = makeStochastic(*P);
  std::cout << nStoch << " out of " << P->n << " nodes already stochastic"
            << std::endl;

  // ~~~~~~~~~~ prepare graph for SG-t-SNE

  // ----- lambda rescaling
  if (params.lambda == 1)
    std::cout << "Skipping λ rescaling..." << std::endl;
  else
    lambdaRescaling(*P, params.lambda, false, params.dropLeaf);

  // ----- symmetrizing
  symmetrizeMatrix(P);

  // ----- normalize matrix (total sum is 1.0)
  matval sum_P = .0;
  for (int i = 0; i < P->nnz; i++) {

    sum_P += P->val[i];
  }
  for (int i = 0; i < P->nnz; i++) {
    P->val[i] /= sum_P;
  }

  // ~~~~~~~~~~ extracting BSDB permutation
  int*   perm = static_cast<int *>(malloc(P->n * sizeof(int)));
  sparse_matrix *Pd = PrepareSparseMatrix(P,perm,params.format, params.method,params.bs);


  // ~~~~~~~~~~ initial embedding coordinates

  dataPoint *y;
  CUDA_CALL(cudaMallocManaged(&y, params.n * params.d * sizeof(dataPoint)));

  if (y_in == NULL) {

    // ----- Initialize Y
    dataPoint *y_rand =generateRandomCoord<dataPoint>(params.n, params.d);
    CUDA_CALL(cudaMemcpy(y, y_rand, params.n * params.d * sizeof(dataPoint),
                         cudaMemcpyHostToDevice));
    free(y_rand);

  } else {
    CUDA_CALL(cudaMemcpy(y, y_in, params.n * params.d * sizeof(dataPoint),
                         cudaMemcpyHostToDevice));
  }


  // ~~~~~~~~~~ gradient descent
  kl_minimization(y, params, *Pd);
  dataPoint *y_return =static_cast<dataPoint *>(malloc(params.n * params.d * sizeof(dataPoint)));

  CUDA_CALL(cudaMemcpy(y_return, y, params.n * params.d * sizeof(dataPoint),
  cudaMemcpyDeviceToHost));



  cudaFree(y);
  free(perm);
  return y_return;
}
template
float *sgtsneCUDA(sparse_matrix* P, tsneparams params, float *y_in,
                  double **timeInfo);
