#include "common.cuh"


coord *generateRandomCoord(uint32_t n, uint32_t d,coord scale) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));
  //srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((coord) rand() / (RAND_MAX))*scale;

  return y;
}

coord* loadData(uint32_t* N,uint32_t* d,char* mode,coord** y_d,uint32_t* perplexity){
	printf("Mode=%s \n",mode);
  coord* y;
  if(strcmp(mode,(char *)"random")==0){

     y=generateRandomCoord(*N,*d,100);
     *perplexity=30;
  }

  if(strcmp(mode,(char *)"batch")==0){

  }
  uint32_t n=*N;
  uint32_t Dim=*d;
  CUDA_CALL(cudaMallocManaged(y_d,Dim*n*sizeof(coord)));

  coord* yc;
  yc=(coord *)malloc(n*Dim*sizeof(coord));
  for(uint32_t i=0;i<n;i++){
    for(uint32_t j=0;j<Dim;j++){
      yc[i+j*n]=y[i*Dim+j];
    }
  }

  CUDA_CALL(cudaMemcpy(*y_d,yc,Dim*n*sizeof(coord), cudaMemcpyHostToDevice));

  free(yc);
  return y;
}
