#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#define MAX_BLOCK_SZ 128

__global__ void orderCheck(uint64_t* d,uint32_t n,int* order){

    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID<n-1){
    if(d[TID+1] <d[TID] ){
      order[blockIdx.x]=1;

    }
  }else{return ;}


}
int issorted(uint64_t * ar, uint32_t n){
for(uint32_t i=0;i<n-1;i++){
  if(ar[i+1]<ar[i]){
    return 0;


  }
}
return 1;

}
int main(){
  unsigned int block_sz = MAX_BLOCK_SZ;
  uint32_t n=1000;
  unsigned int max_elems_per_block = block_sz;
  unsigned int grid_sz = n / max_elems_per_block; //length /blocks
  uint64_t ar[1000];
  for(int i=0;i<1000;i++){
    ar[i]=i;

  }
  printf("issorted=%d\n",issorted(ar,n) );
  uint64_t *ard;
  cudaMalloc(&ard, sizeof(uint64_t) *n);
  cudaMemcpy(ard, ar, sizeof(uint64_t) * n, cudaMemcpyHostToDevice);
  int* order;


  cudaMalloc(&order, grid_sz*sizeof(int) );
  orderCheck<<<grid_sz,block_sz>>>(ard,n,order);
  int outOfOrder=thrust::reduce(thrust::device, order, order + grid_sz, 0);
      if(outOfOrder>0){
        printf("unordered\n" );
      }else{
        printf("skipped\n" );
      }


  return 0;
}
