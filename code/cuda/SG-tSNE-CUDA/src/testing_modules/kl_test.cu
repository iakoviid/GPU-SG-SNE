#include "../gradient_descend.cuh"
#include "../gradient_descend.hpp"
#include "../graph_rescaling.hpp"
#include "../sparsematrix.hpp"
#include "../utils_cuda.cuh"
#include <random>
#include "../prepareMatrix.cuh"
#include "../sparse_reorder.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../types.hpp"

template <class dataPoint>
dataPoint maxerror(dataPoint *const w, dataPoint *dv,int* perm, int n, int d) {

  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  dataPoint maxError = 0;
  dataPoint avgError = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      dataPoint error=fabs(v[i+j*n] - w[perm[i]*d+j]);
         if (error > maxError) {
           maxError =error;

      }
      avgError += error;
    }
  }

  printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d size=%d\n",
         maxError, 1, v[1], w[1 * d], avgError / (n * d), n, n * d);
  free(v);
  return maxError;
}

coord *generateRandomCoord(int n, int d) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));

  for (int i = 0; i < n * d; i++)
    y[i] = ((coord)rand() / (RAND_MAX)) * .0001;

  return y;
}

template <class dataPoint>
void copydata(dataPoint *const w, dataPoint *dw, int n, int d) {
  dataPoint *v = (dataPoint *)malloc(sizeof(dataPoint) * n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {

      v[i + n * j] = w[i * d + j];
    }
  }
  cudaMemcpy(dw, v, d * n * sizeof(dataPoint), cudaMemcpyHostToDevice);
  free(v);
  return;
}
sparse_matrix *generateRandomCSC(int n) {

  sparse_matrix *P = (sparse_matrix *)malloc(sizeof(sparse_matrix));

  P->n = n;
  P->m = n;

  P->col = (matidx *)malloc((n + 1) * sizeof(matidx));

  for (int j = 0; j < n; j++)
    P->col[j] = rand() % 10 + 2;

  int cumsum = 0;
  for (int i = 0; i < P->n; i++) {
    int temp = P->col[i];
    P->col[i] = cumsum;
    cumsum += temp;
  }
  P->col[P->n] = cumsum;
  P->nnz = cumsum;

  P->row = (matidx *)malloc((P->nnz) * sizeof(matidx));
  P->val = (matval *)malloc((P->nnz) * sizeof(matval));

  std::uniform_real_distribution<double> unif(0, 1);
  std::default_random_engine re;

  for (int l = 0; l < P->nnz; l++) {
    P->row[l] = rand() % n;
    P->val[l] = unif(re);
  }

  return P;
}

//nvcc testing_modules/kl_test.cu gradient_descend.cu timers.cpp utils.cu sparse_reorder.cu pq.cu FrepNoReloc.cu Frep.cpp gradient_descendCPU.cpp graph_rescaling.cpp gridding.cpp gridding.cu non_periodic_conv.cpp non_periodic_convD.cu nuconv.cpp nuconv.cu pq.cpp  relocateData.cpp sparsematrix.cpp  -arch=sm_60 -lfftw3 -lcufft -lcusparse -lcusolver
//./sg_test 8381 8381 251430  <pbmc-graph.mtx
int main(int argc, char **argv) {
  srand(time(NULL));

  int n = atoi(argv[1]);
  int d = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  int nz = atoi(argv[4]);
  int format = atoi(argv[5]);
  int bs= atoi(argv[6]);
  char* method= argv[7];

  int N = n;
  int M = n;
  struct timeval t1, t2;
  double elapsedTime;
  int *I, *J;
  double *val;
  I = (int *)malloc(sizeof(int) * nz);
  J = (int *)malloc(sizeof(int) * nz);
  val = (coord *)malloc(sizeof(coord) * nz);
  for (int i = 0; i < nz; i++) {
    scanf("%d %d %lf\n", &J[i], &I[i], &val[i]);
    I[i]--;
    J[i]--;
  }
  sparse_matrix *P = (sparse_matrix *)malloc(sizeof(sparse_matrix));
  P->val = (double *)calloc(nz, sizeof(double));
  P->row = (int *)calloc(nz, sizeof(int));
  P->col = (int *)calloc(M + 1, sizeof(int));

  for (int i = 0; i < nz; i++) {
    P->val[i] = val[i];
    P->row[i] = J[i];
    P->col[I[i] + 1]++;
  }
  for (int i = 0; i < M; i++) {
    P->col[i + 1] += P->col[i];
  }
  P->n = N;
  P->m = M;
  P->nnz = nz;

  tsneparams params;
  params.d = d;
  params.n = n;
  params.alpha = 12;
  params.maxIter = iterations;
  params.earlyIter = iterations / 4;
  params.np = 1;
  gettimeofday(&t1, NULL);
  coord *y, *y_d;
  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);

  uint32_t nStoch = makeStochastic(*P);
  std::cout << nStoch << " out of " << P->n << " nodes already stochastic"<< std::endl;
  // lambdaRescaling(*P, params.lambda, false, params.dropLeaf);

  symmetrizeMatrix(P);

  double sum_P = .0;
  for (int i = 0; i < P->nnz; i++) {

    sum_P += P->val[i];
  }
  for (int i = 0; i < P->nnz; i++) {
    P->val[i] /= sum_P;
  }

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

  int*   perm = static_cast<int *>(malloc(P->n * sizeof(int)));


  //kl_minimizationCPU(y, params, *P);
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  double timecpu = elapsedTime;
  printf("timecpu=%lf\n",timecpu );
  params.d = d;
  params.n = n;
  params.alpha = 12;
  params.maxIter = iterations;
  params.earlyIter = iterations / 4;
  params.np = 1;

  gettimeofday(&t1, NULL);
  sparse_matrix *Pd = PrepareSparseMatrix(P,perm,format, method, bs);
  kl_minimization(y_d, params, *Pd);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  double timegpu = elapsedTime;
  printf("Speedup=%lf\n", timecpu / timegpu);
  printf("timegpu=%lf\n",timegpu );
  //maxerror(y, y_d,perm, n, d);
  coord *y_copy =static_cast<coord *>(malloc(params.n * params.d * sizeof(coord)));

  CUDA_CALL(cudaMemcpy(y_copy, y_d, params.n * params.d * sizeof(coord),cudaMemcpyDeviceToHost));

  extractEmbeddingTextT(y_copy, params.n, params.d, "gpuEmbedding.txt");
  //extractEmbeddingText(y, params.n, params.d, "cpuEmbedding.txt");

  cudaFree(y_d);
  free(y);
  free(I);
  free(J);
  free(val);
  return 0;
}
