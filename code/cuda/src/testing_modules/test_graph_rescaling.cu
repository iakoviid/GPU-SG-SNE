#include <iostream>
#include <random>
#include "../common.cuh"
#include "../types.hpp"
#include "../sparsematrix.hpp"
#include "../graph_rescaling.hpp"
#include "../graph_rescaling.cuh"
#include "../sparsematrix.cuh"


//#include <metis.h>

#define N_NUM 1
#define D_NUM 1
#define H_NUM 3

double * generateRandomCoord( int n, int d ){

  double *y = (double *) malloc( n*d*sizeof(double) );

  std::uniform_real_distribution<double> unif(0,1);
  std::default_random_engine re;

  for (int i=0; i<n*d; i++)
    y[i] = unif(re);

  return y;

}

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
             int same,double maxErr) {
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
        if (abs(w[i + j * n] - v[i + j * n]) < maxErr ){
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else {
          bro = 0;
          std::cout << "Error "
               << "Host=" << w[i + j * n] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << std::endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
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

bool testAttractiveTerm( int n, int d){

  bool flag = true;

  double *y  = generateRandomCoord( n, d );
  sparse_matrix *P = generateRandomCSC(n);
  symmetrizeMatrix( P );

  coord *yd;
  matval *vald;
  matidx *cold,*rowd;
  int nnz=P->nnz;
  CUDA_CALL(cudaMallocManaged(&yd, (d) * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&cold, nnz * sizeof(matidx)));
  CUDA_CALL(cudaMallocManaged(&vald,nnz * sizeof(matval)));
  CUDA_CALL(cudaMallocManaged(&rowd, (n+1) * sizeof(matidx)));

  cudaMemcpy(cold,  P->row,  nnz * sizeof(matidx), cudaMemcpyHostToDevice);
  cudaMemcpy(vald,  P->val, nnz * sizeof(matval), cudaMemcpyHostToDevice);
  cudaMemcpy(rowd,  P->col, (n+1) * sizeof(matidx), cudaMemcpyHostToDevice);

  copydata(y, yd, n, d);


  //lambdaRescaling( *P, 12, true, true );
  sparse_matrix *Pd=(sparse_matrix *) malloc(sizeof(sparse_matrix));
  Pd->val=vald;
  Pd->col=cold;
  Pd->row=rowd;
  Pd->n=n;
  Pd->nnz=nnz;
  //lambdaRescalingGPU( *Pd, 12, true, true );
  uint32_t nStoch = makeStochastic( *P );
 uint32_t gpns= makeStochasticGPU( Pd);
  compair( P->val, Pd->val, nnz,1, "λ", 1,0.01);



  //deallocate(csb);
  free( P );
  cudaFree( vald );
  cudaFree( cold );
  cudaFree( rowd );
  free(Pd);
  cudaFree(yd);

  free(y);
  return flag;

}


int main(int argc, char **argv)
{

  int n= atoi(argv[1]);
  int d = atoi(argv[2]);

  std::cout << "\n\n *** TESTING SG-TSNE-PI INSTALLATION ***\n\n";

  std::cout << "\n - Attractive term [PQ]\n";



      bool status = testAttractiveTerm(n, d);


return 0;
}
