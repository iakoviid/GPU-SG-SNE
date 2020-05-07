#include "main.cuh"



int main(int argc, char **argv){
  uint32_t Dim=atoi(argv[1]);
  uint32_t N=1<<atoi(argv[2]);
  uint32_t d=atoi(argv[3]);
  uint32_t perplexity;

  coord* x_d;
  coord* x=loadData(&N,&Dim,(char *)"random",&x_d,&perplexity);

  printf("dimension= %d \n",Dim );
  printf("N= %d\n",N );



  coord* y;
  y=generateRandomCoord(N,d,0.001);
  coord* y_d;
  coord* yc;
  for(int i=0;i<N;i++){
    for(int j=0;j<d;j++){
      yc[i+j*N]=y[i*d+j];
    }
  }
  CUDA_CALL(cudaMallocManaged(&y_d,d*N*sizeof(coord)));
  CUDA_CALL(cudaMemcpy(y_d,yc,d*N*sizeof(coord), cudaMemcpyHostToDevice));

  tsneRun(y_d,x_d,N,d,Dim,perplexity);
  tsneRunCpu(y,x,N,d,Dim,perplexity);

  extractEmbeddingText(y,N,d);



  free(y);
  free(x);
  CUDA_CALL(cudaFree(y_d));
  CUDA_CALL(cudaFree(x_d));
  return 0;
}
