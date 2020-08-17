#include "sparsematrix.cuh"

/*

void free_sparse_matrix(sparse_matrix * P){
  free(P->row);
  free(P->col);
  free(P->val);

}
*/
void free_sparse_matrixGPU(sparse_matrix * P){

  CUDA_CALL(cudaFree(P->row));
  CUDA_CALL(cudaFree(P->col));
  CUDA_CALL(cudaFree(P->val));

}

template <class T>
__device__ T warp_reduce(T val){
  for(int offset=32/2;offset>0;offset/=2){
    val+=__shfl_down_sync(FULL_WARP_MASK,val,offset);

  }
  return val;
}

__global__ void makeStochasticKernel(matval val, matidx col,matidx row, uint32_t n, uint32_t* stoch ){
  const uint32_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t warp_id=thread_id/32;
  const uint32_t lane=thread_id%32;

  const uint32_t j=warp_id;
  if(j<n){
    matval sum=0;
    for ( uint32_t t = col [j] ; t < col[j+1] ; t+=32){
      sum+=val[t];
    }
    warp_reduce(sum);
    __shfl_sync(FULL_MASK, sum,0);
    if(fabs(sum-1)>1e-12){
      for ( uint32_t t = col [j] ; t < col[j+1] ; t+=32){
        val[t]/=sum;
      }
      if(lane==0){
        stoch[j]=0;
      }
    }else{
      if(lane==0){
        stoch[j]=1;
      }
    }


  }



}

uint32_t makeStochasticGPU(sparse_matrix *P){

  uint32_t * stoch;
  CUDA_CALL(cudaMallocManaged(&stoch, n * sizeof(uint32_t)));


  makeStochasticKernel<<<32,256>>>(P->val,P->col,P->row,n,stoch);

  thrust::device_ptr<coord> stoch_ptr(stoch);
  uint32_t nStoch=thrust::reduce(stoch_ptr,stoch_ptr+P->n);


  CUDA_CALL(cudaFree(stoch));
  return nStoch;

}


void  permuteMatrixGPU(sparse_matrix *P, int *perm, int *iperm) {
  // Get sparse matrix
  matidx* row_P = P->row;
  matidx* col_P = P->col;
  matval* val_P = P->val;

  int N = P->n; matidx nnz = P->nnz;

  // Allocate memory for permuted matrix
  matval* perm_val_P; = (matval*) malloc( nnz    * sizeof(matval));
  CUDA_CALL(cudaMallocManaged(&perm_val_P, nnz * sizeof(matval)));

  size_t pBufferSizeInBytes = 0;
  void *pBuffer = NULL;

  // step 1: allocate buffer
  cusparseXcsrsort_bufferSizeExt(handle, N, N, nnz, col_P, row_P, &pBufferSizeInBytes);
  cudaMalloc( &pBuffer, sizeof(char)* pBufferSizeInBytes);

  //step 3: sort CSR format
  cusparseXcsrsort(handle, N, N, nnz, descrA, col_P, row_P, perm, pBuffer);


  cusparseDgthr(handle, nnz, val_P, perm_val_P, perm, CUSPARSE_INDEX_BASE_ZERO);


  cudaFree(P->val); P->val = perm_val_P;

}
