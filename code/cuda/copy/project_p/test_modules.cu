#include <iostream>
#include <random>
#include "common.cuh"
#include "pq.hpp"
#include "pq.cuh"
#include "types.hpp"
#include "sparsematrix.hpp"
#include "graph_rescaling.hpp"
#include "graph_rescaling.cuh"

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


  double *Fg = (double *) calloc( n*d , sizeof(double) );
  double *Ft = (double *) calloc( n*d , sizeof(double) );

  pq( Fg, y, P->val, P->row, P->col, n, d);
  coord *Ftd,*yd;
  matval *vald;
  matidx *cold,*rowd;
  int nnz=P->nnz;
  CUDA_CALL(cudaMallocManaged(&Ftd, (d) * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&yd, (d) * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&cold, (n+1) * sizeof(matidx)));
  CUDA_CALL(cudaMallocManaged(&vald,nnz * sizeof(matval)));
  CUDA_CALL(cudaMallocManaged(&rowd, nnz * sizeof(matidx)));

  cudaMemcpy(cold,  P->col,  (n+1) * sizeof(matidx), cudaMemcpyHostToDevice);
  cudaMemcpy(vald,  P->val, nnz * sizeof(matval), cudaMemcpyHostToDevice);
  cudaMemcpy(rowd,  P->row, nnz * sizeof(matidx), cudaMemcpyHostToDevice);

  copydata(y, yd, n, d);



  PQKernel<<<32,256>>>(Ftd, yd,vald , cold, rowd, n, d);

  compair(Fg, Ftd, n, d, "PQ", 0,0.01);

  lambdaRescaling( *P, 12, false, true );
  sparse_matrix *Pd=(sparse_matrix *) malloc(sizeof(sparse_matrix));
  Pd->val=vald;
  Pd->col=cold;
  Pd->row=rowd;
  Pd->n=n;
  Pd->nnz=nnz;
  lambdaRescalingGPU( *Pd, 12, false, true );
  compair( P->val, vald, nnz,1, "λ", 1,0.01);



  //deallocate(csb);
  free( P );
  //cudaFree( vald );
//  cudaFree( cold );
  //cudaFree( rowd );
  //free(Pd);
  cudaFree(yd);
  cudaFree(Ftd);

  free(y);
  free(Fg);
  free(Ft);
  return flag;

}


int main(void)
{

  int n[N_NUM] = {2};
  int d[D_NUM] = {3};

  std::cout << "\n\n *** TESTING SG-TSNE-PI INSTALLATION ***\n\n";

  std::cout << "\n - Attractive term [PQ]\n";

  int n_pass = 0;
  int n_fail = 0;

  for (int i = 0; i < N_NUM; i++){
    for (int j = 0; j < D_NUM; j++){
      //std::cout << "   > N = " << n[i] << " D = " << d[j] << "..." << std::flush;

      bool status = testAttractiveTerm(n[i], d[j]);
      n_pass +=  status;
      n_fail += !status;
/*
      if ( status )
        std::cout << "PASS" << std::endl;
      else
        std::cout << "FAIL!!!" << std::endl;

    */
  }
  }
/*
  std::cout << "\n\n *** SUMMARY ***\n";
  std::cout << "  > " << n_pass << " tests passed" << std::endl;
  std::cout << "  > " << n_fail << " tests failed" << std::endl;

  if (n_fail == 0){
    std::cout << "\n *** INSTALLATION SUCESSFUL ***" << std::endl << std::endl;
    return 0;
  } else {
    std::cout << "\n *** INSTALLATION FAILED!!! ***" << std::endl << std::endl;
    return -1;
  }
*/
}
