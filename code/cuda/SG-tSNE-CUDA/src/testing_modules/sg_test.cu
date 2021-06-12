#include "common.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../types.hpp"
#include "../sgtsne.cuh"
#include "../utils.cuh"
#include <fstream>

//!./src/sg_test 60000 2 1000 7507854 0 32 metis <knnMnist.txt
int main(int argc, char **argv) {
  //srand(time(NULL));

  int n = atoi(argv[1]);
  int d = atoi(argv[2]);
  int iterations = atoi(argv[3]);
  int nz = atoi(argv[4]);
  int format = atoi(argv[5]);
  int bs= atoi(argv[6]);
  char* method= argv[7];
  int error=atoi(argv[8]);
  int ng=atoi(argv[9]);

  int N = n;
  int M = n;
  int *I, *J;
  matval *val;
  I = (int *)malloc(sizeof(int) * nz);
  J = (int *)malloc(sizeof(int) * nz);
  val = (coord *)malloc(sizeof(coord) * nz);
  for (int i = 0; i < nz; i++) {
    scanf("%d %d %f\n", &J[i], &I[i], &val[i]);
    I[i]--;
    J[i]--;
  }

  //ReadMatrix(&M,&N,&nz,&I,&J, &val,argc,argv);
  sparse_matrix<coord> *P = (sparse_matrix<coord> *)malloc(sizeof(sparse_matrix<coord>));
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

  tsneparams params;
  params.d = d;
  params.n = n;
  params.alpha = 12;
  params.maxIter = iterations;
  params.earlyIter = iterations / 4;
  params.np = 1;
  params.format=format;
  params.method=method;
  params.bs=bs;
  params.lambda=1;
  params.h=0;
  params.ComputeError=error;
  params.ng=ng;

  struct timeval t1, t2;
  coord *yin=(coord*)malloc(sizeof(coord)*d*n);
  ifstream yinf;

  if(d==2){
  yinf.open("yin2d.txt");
}else if(d==3){
   yinf.open("yin3d.txt");
}
  for(int i=0;i<n;i++){
    for(int j=0;j<d;j++){
    yinf>>yin[i+j*n];
  }
  }
  yinf.close();

  gettimeofday(&t1, NULL);
  coord* y =sgtsneCUDA<coord>( P,  params,yin,NULL);
  gettimeofday(&t2, NULL);
  double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("time=%lf\n",elapsedTime );

  extractEmbeddingTextT(y, params.n, params.d, "sgtsneEmbedding.txt");
  free(yin);
  free( y );


return 0;
}
