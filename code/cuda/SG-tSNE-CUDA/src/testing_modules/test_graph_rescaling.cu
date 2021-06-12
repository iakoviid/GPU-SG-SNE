#include <iostream>
#include <random>
#include "../common.cuh"
#include "../types.hpp"
#include "../sparsematrix.hpp"
#include "../graph_rescaling.hpp"
#include "../graph_rescaling.cuh"

#include <sys/time.h>

matval * generateRandomCoord( int n, int d ){

  matval *y = (matval *) malloc( n*d*sizeof(matval) );

  std::uniform_real_distribution<matval> unif(0,1);
  std::default_random_engine re;

  for (int i=0; i<n*d; i++)
    y[i] = unif(re);

  return y;

}

sparse_matrix<matval> *generateRandomCSC(int n){

  sparse_matrix<matval>*P = (sparse_matrix<matval> *) malloc(sizeof(sparse_matrix<matval>));

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

  std::uniform_real_distribution<matval> unif(0,1);
  std::default_random_engine re;

  for (int l = 0; l < P->nnz; l++){
    P->row[l] = rand() % n;
    P->val[l] = unif(re);
  }

  return P;

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
template <class dataPoint>
void compair(dataPoint *const w, dataPoint *dv, int n, int d,const  char *message,
             int same,matval maxErr) {
  int bro = 1;
  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  printf(
      "----------------------------------Compair %s----------------------------"
      "-----------\n",
      message);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if (same == 0) {
        if(i<10){
        printf("%f vs %f\n",w[i * d + j], v[i + j * n] );}
        if (abs(w[i * d + j] - v[i + j * n]) < maxErr) {
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else {
          bro = 0;
          std::cout << "Error "
               << "Host=" << w[i * d + j] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << std::endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      } else {
        if(i<10){
        printf("%f vs %f\n",w[i + j * n], v[i + j * n] );}
        if (abs(w[i + j * n] - v[i + j * n]) < maxErr ){
        } else {
          bro = 0;
          std::cout << "Error "
               << "Host=" << w[i + j * n] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << std::endl;
        }
      }
    }
  }
  if (bro == 1) {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Success~~~~~~~~~~~~~~~~~~~~~~~~\n");
  } else {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Error~~~~~~~~~~~~~~~~~~~~~~~~\n");
  }
  free(v);
}

bool testAttractiveTerm( int n,int nz){

  bool flag = true;
  int N = n;
  int M = n;
  int *I, *J;
  matval *val;
  I = (int *)malloc(sizeof(int) * nz);
  J = (int *)malloc(sizeof(int) * nz);
  val = (coord *)malloc(sizeof(coord) * nz);
  for (int i = 0; i < nz; i++) {
    scanf("%d %d %f\n", &J[i], &I[i], &val[i]);
    I[i];
    J[i];
  }
  sparse_matrix<matval> *P = (sparse_matrix<matval> *)malloc(sizeof(sparse_matrix<matval>));
  P->val = (matval *)calloc(nz, sizeof(matval));
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

  //symmetrizeMatrix( P );

  int nnz=P->nnz;
  sparse_matrix<matval> *Pd=(sparse_matrix<matval> *) malloc(sizeof(sparse_matrix<matval>));
  CUDA_CALL(cudaMallocManaged(&(Pd->col), nnz * sizeof(matidx)));
  CUDA_CALL(cudaMallocManaged(&(Pd->val),nnz * sizeof(matval)));
  CUDA_CALL(cudaMallocManaged(&(Pd->row), (n+1) * sizeof(matidx)));

  cudaMemcpy((Pd->col),  P->row,  nnz * sizeof(matidx), cudaMemcpyHostToDevice);
  cudaMemcpy((Pd->val),  P->val, nnz * sizeof(matval), cudaMemcpyHostToDevice);
  cudaMemcpy((Pd->row),  P->col, (n+1) * sizeof(matidx), cudaMemcpyHostToDevice);

  Pd->n=n;
  Pd->m=n;
  Pd->nnz=nnz;
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  uint32_t nStoch = makeStochastic( *P );
  //lambdaRescaling( *P, 12, true, true );
  gettimeofday(&t2, NULL);
  double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("GraphCPU=%lf\n",elapsedTime );

  gettimeofday(&t1, NULL);
  uint32_t gpns= makeStochasticGPU( Pd);
  cudaDeviceSynchronize();
  //lambdaRescalingGPU( *Pd, 12, true, true );
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("GraphGPU=%lf\n",elapsedTime );

  printf("Stockcpu=%d vs Stockgpu=%d \n",nStoch,gpns );
  compair( P->val, Pd->val, nnz,1, "λ", 1,0.00001);

  //symmetrizeMatrixGPU(Pd,handle);


  //deallocate(csb);
  free( P );
  free(Pd);
  free(I);
  free(J);
  free(val);


  return flag;

}


int main(int argc, char **argv)
{

  int n= atoi(argv[1]);
  int nz = atoi(argv[2]);
  bool status = testAttractiveTerm(n, nz);


return 0;
}
