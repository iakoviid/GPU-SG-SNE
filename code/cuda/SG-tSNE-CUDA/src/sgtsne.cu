/*!
  \file   sgtsne.cpp
  \brief  Entry point to SG-t-SNE

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.


*/
#include "graph_rescaling.cuh"
#include "prepareMatrix.cu"
#include "sgtsne.cuh"
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <fstream>
template <class dataPoint>
void free_sparse_matrixGPU(sparse_matrix<dataPoint> *P) {

  gpuErrchk(cudaFree(P->row));
  gpuErrchk(cudaFree(P->col));
  gpuErrchk(cudaFree(P->val));
}

template <class dataPoint>
dataPoint *sgtsneCUDA(sparse_matrix<dataPoint> *P, tsneparams params,
                      dataPoint *y_in, double *timeInfo) {
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

  printParams(params);

  // ~~~~~~~~~~ make sure input matrix is column stochastic
//  uint32_t nStoch = makeStochasticGPU(P);
 // std::cout << nStoch << " out of " << P->n << " nodes already stochastic" << std::endl;

  // ~~~~~~~~~~ prepare graph for SG-t-SNE

  // ----- lambda rescaling
  if (params.lambda == 1)
    std::cout << "Skipping λ rescaling..." << std::endl;
  else
    lambdaRescalingGPU(*P, params.lambda, false, params.dropLeaf);

  // ----- symmetrizing
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  sparse_matrix<dataPoint> *Pd =
      (sparse_matrix<dataPoint> *)malloc(sizeof(sparse_matrix<dataPoint>));

cudaDeviceSynchronize();
  int* row;
  Pd->nnz=SymmetrizeMatrix(handle, &(Pd->val),&(row),
                   &(Pd->col), P->val, P->col, P->n, P->row,P->nnz);
  Pd->n=P->n;
  Pd->m=P->n;
cudaDeviceSynchronize();


  // ----- normalize matrix (total sum is 1.0)
//  dataPoint sum_P = thrust::reduce(Pd->val, Pd->val+Pd->nnz);
 // ArrayScale<<<64, 1024>>>(Pd->val, 1 / sum_P, Pd->nnz);

  // ~~~~~~~~~~ extracting BSDB permutation
  double elapsedTime;
  struct timeval t1, t2;

  gettimeofday(&t1, NULL);

  if(params.format==2){
  cudaMalloc((void **)&(Pd->row), sizeof(int) * (Pd->nnz));
  Csr2Coo(Pd->nnz,Pd->n,row,Pd->col,Pd->row);
  Pd->format=params.format;
}
  else if(params.format==3){
    Pd->row=row;
    PrepareHybrid(Pd);
    Pd->format=params.format;
    cudaFree(Pd->col);
    cudaFree(Pd->val);

  }
  cudaFree(row);
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  timeInfo[7*1000+1]=elapsedTime;
  std::cout << "nnz= "<< Pd->nnz << std::endl;

  // ~~~~~~~~~~ initial embedding coordinates
  dataPoint *y;
  gpuErrchk(cudaMallocManaged(&y, params.n * params.d * sizeof(dataPoint)));

  if (y_in == NULL) {

    // ----- Initialize Y
    dataPoint *y_rand = generateRandomGaussianCoord<dataPoint>(params.n, params.d);
    gpuErrchk(cudaMemcpy(y, y_rand, params.n * params.d * sizeof(dataPoint),
                         cudaMemcpyHostToDevice));
    free(y_rand);

  } else {
    gpuErrchk(cudaMemcpy(y, y_in, params.n * params.d * sizeof(dataPoint),
                         cudaMemcpyHostToDevice));
  }

  // ~~~~~~~~~~ gradient descent
  kl_minimization(y, params, *Pd,timeInfo);
  dataPoint *y_return =
      static_cast<dataPoint *>(malloc(params.n * params.d * sizeof(dataPoint)));

  gpuErrchk(cudaMemcpy(y_return, y, params.n * params.d * sizeof(dataPoint),
                       cudaMemcpyDeviceToHost));

  // ~~~~~~~~~~ dellocate memory
  cudaFree(y);
  return y_return;
}
template float *sgtsneCUDA(sparse_matrix<float> *P, tsneparams params,
                           float *y_in, double *timeInfo);
