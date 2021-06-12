
#include <stdio.h>
#include <stdlib.h>
#include "../types.hpp"
//#include "../sgtsne.hpp"
#include "../sgtsne.cuh"

int main(int argc, char *argv[]) {
  int M, N, nz;
  int *I, *J;
  double *val;
  // ReadMatrix(&M,&N,&nz,&I,&J, &val,argc,argv);
  N = atoi(argv[1]);
  M = atoi(argv[2]);
  nz = atoi(argv[3]);
  I = (int *)malloc(sizeof(int) * nz);
  J = (int *)malloc(sizeof(int) * nz);
  val = (coord *)malloc(sizeof(coord) * nz);

  for (int i = 0; i < nz; i++) {
    scanf("%d %d %lf\n", &J[i], &I[i], &val[i]);
    I[i]--;
    J[i]--;
  }
  sparse_matrix P;
   P.val = (double *)calloc(nz, sizeof(double));
   P.row = (int *)calloc(nz, sizeof(int));
   P.col = (int *)calloc(M + 1, sizeof(int));

  for (int i = 0; i < nz; i++) {
    P.val[i] = val[i];
    P.row[i] = J[i];
    P.col[I[i] + 1]++;
  }
  for (int i = 0; i < M; i++) {
    P.col[i + 1] += P.col[i];
  }
  P.n=N;
  P.m=M;
  P.nnz=nz;
  tsneparams params;
  params.n=N;
  params.maxIter=1000;
  params.d=2;
  double *y=sgtsneCUDA(P, params,NULL,NULL);
  extractEmbeddingText(y,params.n,params.d);
  free(I);
  free(J);
  free(val);

  return 0;


}
