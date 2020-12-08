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
#define Blocksize 512

__global__ void makeStochasticKernel(matval val, matidx col,matidx row, uint32_t n, uint32_t* stoch ){
  const uint32_t thread_id=blockIdx.x*blockDim.x+threadIdx.x;
  const uint32_t warp_id=thread_id/32;
  const uint32_t lane=thread_id%32;
  __shared__ matval sdata[Blocksize/32];
  const unsigned int n_warps=gridDim.x*blockDim.x/32;
  const unsigned int warp_id_block=threadIdx.x/32;

  const uint32_t j=warp_id;
  for (;j < n; j=j+n_warps) {
    matval sum=0;
    for ( uint32_t t = row[j] ; t < row[j+1] ; t+=32){
      sum+=val[t];
    }
    warp_reduce(sum);
    if(lane==0){sdata[warp_id_block]=sum;}
    __syncwarp(FULL_WARP_MASK);
    sum=sdata[warp_id_block];

    if(fabs(sum-1)>1e-12){
      for ( uint32_t t = row [j] ; t < row[j+1] ; t+=32){
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


  makeStochasticKernel<<<32,Blocksize>>>(P->val,P->col,P->row,n,stoch);

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
void add_cusparse(sparse_matrix A,sparse_matrix B,sparse_matrix *C){
  // --- Initialize cuSPARSE
  cusparseHandle_t handle;    cusparseCreate(&handle);
  // --- Initialize matrix descriptors
  cusparseMatDescr_t descrA, descrB, descrC;
  cusparseSafeCall(cusparseCreateMatDescr(&descrA));
  cusparseSafeCall(cusparseCreateMatDescr(&descrB));
  cusparseSafeCall(cusparseCreateMatDescr(&descrC));
  matidx M=A.m;
  matidx N=A.n;
  // --- Summing the two matrices
  matidx baseC, nnz3;
  // --- nnzTotalDevHostPtr points to host memory
  matidx *nnzTotalDevHostPtr = &nnz3;
  cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
  matidx *d_csrRowPtrC;
  cudaMalloc(&d_csrRowPtrC, (M + 1) * sizeof(matidx));
  cusparseXcsrgeamNnz(handle, M, N, descrA, A.nnz, A.row, A.col, descrB, B.nnz, B.row, B.col, descrC, d_csrRowPtrC, nnzTotalDevHostPtr);
  if (NULL != nnzTotalDevHostPtr) {
      nnz3 = *nnzTotalDevHostPtr;
  }
  else{
      cudaMemcpy(&nnz3, d_csrRowPtrC + M, sizeof(matidx), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, d_csrRowPtrC, sizeof(matidx), cudaMemcpyDeviceToHost);
      nnz3 -= baseC;
  }
  matidx *d_csrColIndC;   cudaMalloc(&d_csrColIndC, nnz3 * sizeof(matidx));
  matval *d_csrValC;    cudaMalloc(&d_csrValC, nnz3 * sizeof(matval));
  matval alpha = 1.f, beta = 1.f;
  cusparseScsrgeam(handle, M, N, &alpha, descrA, A.nnz, A.val, A.row, A.col, &beta, descrB, B.nnz, B.val, B.row, B.col, descrC, d_csrValC, d_csrRowPtrC, d_csrColIndC);

  C->row=d_csrRowPtrC;
  C->nnz=nnz3;
  C->col=d_csrColIndC;
  C->val=d_csrValC;
  C->m=M;
  C->n=N;

}

void symmetrizeMatrix(sparse_matrix*P){
  matidx* csc_row_ptr_at;
  cudaMalloc(reinterpret_cast<void**>(&csc_row_ptr_at),
          (P->nnz)*sizeof(matidx));
  matidx* csc_column_ptr_at;
  cudaMalloc(reinterpret_cast<void**>(&csc_column_ptr_at),
          (P->n+1)*sizeof(matidx));
  matval* csc_values_at;
  cudaMalloc(reinterpret_cast<void**>(&csc_values_at),
          (P->nnz)*sizeof(matval));
  cusparseHandle_t handle;
 cusparseCreate(&handle);
  // Do the transpose operation
  cusparseScsr2csc(handle, P->m, P->n,
                  P->nnz, P->val, P->row,
                   P->col, csc_values_at, csc_row_ptr_at,
                   csc_column_ptr_at, CUSPARSE_ACTION_NUMERIC,
                   CUSPARSE_INDEX_BASE_ZERO);
  cudaDeviceSynchronize();
  sparse_matrix B;
  B.m=P->m;
  B.n=P->n;
  B.row=csc_column_ptr_at;
  B.col=csc_row_ptr_at;
  B.val=csc_values_at;
  sparse_matrix C;

  add_cusparse( *P, B,& C);
  cudaDeviceSynchronize();

  cudaFree(csc_column_ptr_at);
  cudaFree(csc_row_ptr_at);
  cudaFree(csc_values_at);

  matval* h_val=(matval *)malloc(sizeof(matval)*C.nnz);
  cudaMemcpy(h_val,C.val,sizeof(matval)*C.nnz,cudaMemcpyDeviceToHost);

  P->row=C.row;
  P->col=C.col;
  P->val=C.val;
  P->nnz=C.nnz;
}
