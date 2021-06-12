#include "perplexity_search0.cuh"
#include "perplexity_search_utils.cuh"
#include <thrust/device_ptr.h>
/*
    Compute the unnormalized pij matrix given a squared distance matrix and a
   target perplexity. pij = exp(-beta * dist ** 2)

    Note that FAISS returns the first row as the same point, with distance = 0.
   pii is defined as zero.
*/

struct FunctionalEntropy {
  __host__ __device__ float operator()(const float &x) const {
      float val = x*log(x);
      return (val != val || isinf(val)) ? 0 : val;
    }
};
__global__ void ComputePijKernel(volatile float *__restrict__ pij,
                                 const float *__restrict__ squared_dist,
                                 const float *__restrict__ betas,
                                 const unsigned int num_points,
                                 const unsigned int num_near_neighbors) {
  register int TID, i, j;
  register float dist, beta;

  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points * num_near_neighbors)
    return;

  i = TID / num_near_neighbors;
  j = TID % num_near_neighbors;

  beta = betas[i];
  dist = squared_dist[TID+i+1];

  // condition deals with evaluation of pii
  // FAISS neighbor zero is i so ignore it
  pij[TID] = (j == 0 & dist == 0.0f) ? 0.0f : __expf(-beta * dist);
}

__global__ void PerplexitySearchKernel(
    volatile float *__restrict__ betas,
    volatile float *__restrict__ lower_bound,
    volatile float *__restrict__ upper_bound, volatile int *__restrict__ found,
    const float *__restrict__ neg_entropy, const float *__restrict__ row_sum,
    const float perplexity_target, const float epsilon, const int num_points) {
  register int i, is_found;
  register float perplexity, neg_ent, sum_P, perplexity_diff, beta, min_beta,
      max_beta;
  i = threadIdx.x + blockIdx.x * blockDim.x;
  if (i >= num_points)
    return;

  neg_ent = neg_entropy[i];
  sum_P = row_sum[i];
  beta = betas[i];

  min_beta = lower_bound[i];
  max_beta = upper_bound[i];

  perplexity = (neg_ent / sum_P) + __logf(sum_P);
  perplexity_diff = perplexity - __logf(perplexity_target);
  is_found = (perplexity_diff < epsilon && -perplexity_diff < epsilon);
  if (!is_found) {
    if (perplexity_diff > 0) {
      min_beta = beta;
      beta = (max_beta == FLT_MAX || max_beta == -FLT_MAX)
                 ? beta * 2.0f
                 : (beta + max_beta) / 2.0f;
    } else {
      max_beta = beta;
      beta = (min_beta == -FLT_MAX || min_beta == FLT_MAX)
                 ? beta / 2.0f
                 : (beta + min_beta) / 2.0f;
    }
    lower_bound[i] = min_beta;
    upper_bound[i] = max_beta;
    betas[i] = beta;
  }
  found[i] = is_found;
}
__global__ void  knn_graph_indices_warp(int * row,int * col,int* I,const int n,const int nn) {
  const uint32_t thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t warp_id = thread_id / 32;
  register const uint32_t lane = thread_id % 32;
  const unsigned int n_warps=gridDim.x*blockDim.x/32;
  register int index;
  for (register uint32_t j=warp_id;j < n;j=j+n_warps) {
    index=nn*j;
    if(lane==0){
      row[j]=index;
    }
    for (uint32_t idx = lane; idx < nn; idx += 32) {
      col[index+idx]=I[ index+j + idx + 1 ];
    }
  }
  if(thread_id==0){row[n]=nn*n;}

}
__global__ void knn_graph_indices(int * row,int * col,int* I,const int n,const int nn){
  register int index;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n ;
       TID += gridDim.x * blockDim.x) {
         index=nn*TID;
         row[TID]=index;
         for(register int idx=0;idx<nn;idx++){
           col[index+idx]=I[ index+TID + idx + 1 ];
         }
         if(TID==n-1){row[n]=nn*n;}

         }
}
template <class dataPoint>
sparse_matrix<dataPoint> perplexityEqualization(int *I, dataPoint *D, int num_points,
                       int num_near_neighbors, dataPoint perplexity_target) {
  sparse_matrix<dataPoint> P;
  dataPoint *val;
  matidx *row, *col;
  CUDA_CALL(cudaMallocManaged(&val,num_near_neighbors * num_points * sizeof(dataPoint)));
  CUDA_CALL(cudaMallocManaged(&col, num_near_neighbors * num_points * sizeof(matidx)));
  CUDA_CALL(cudaMallocManaged(&row, (num_points+1) * sizeof(matidx)));
  thrust::device_ptr<dataPoint> val_ptr(val);
  thrust::device_vector<dataPoint> pij(num_near_neighbors*num_points);
  thrust::device_vector<dataPoint> squared_dist(D,D+(1+num_near_neighbors)*num_points);
  cublasHandle_t handle;
  cublasCreate(&handle);
  knn_graph_indices<<<64,1024>>>(row,col,I,num_points,num_near_neighbors);
  SearchPerplexity(handle, pij,squared_dist,perplexity_target, 1e-5,num_points,num_near_neighbors);
 // std::cout<<"sparse_matrix: \n";
 // for(int i=0;i<91;i++){std::cout<<pij[i]<<"\n";}
  thrust::copy(pij.begin(), pij.end(),val_ptr);
  P.n   = num_points;
  P.m   = num_points;
  P.nnz = num_points * num_near_neighbors;
  P.row = row;
  P.col = col;
  P.val = val;
cudaDeviceSynchronize();

  return P;
}
template <class dataPoint>
  __global__ void addScalarKernel(dataPoint *a, dataPoint scalar, uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] += scalar;
  }
}

void SearchPerplexity(cublasHandle_t &handle, thrust::device_vector<float> &pij,
                 thrust::device_vector<float> &squared_dist,
                 const float perplexity_target, const float epsilon,
                 const int num_points, const int num_near_neighbors) {
  // use beta instead of sigma (this matches the bhtsne code but not the paper)
  // beta is just multiplicative instead of divisive (changes the way binary
  // search works)
  thrust::device_vector<float> betas(num_points, 1.0f);
  thrust::device_vector<float> lower_bound_beta(num_points, -FLT_MAX);
  thrust::device_vector<float> upper_bound_beta(num_points, FLT_MAX);
  thrust::device_vector<float> entropy(num_points * num_near_neighbors);
  thrust::device_vector<int> found(num_points);

  // TODO: this doesn't really fit with the style
  const int BLOCKSIZE1 = 1024;
  const int NBLOCKS1 = iDivUp(num_points * num_near_neighbors, BLOCKSIZE1);

  const int BLOCKSIZE2 = 128;
  const int NBLOCKS2 = iDivUp(num_points, BLOCKSIZE2);

  size_t iters = 0;
  int all_found = 0;
  thrust::device_vector<float> row_sum, neg_entropy;
  do {
    // compute Gaussian Kernel row
    ComputePijKernel<<<NBLOCKS1, BLOCKSIZE1>>>(
        thrust::raw_pointer_cast(pij.data()),
        thrust::raw_pointer_cast(squared_dist.data()),
        thrust::raw_pointer_cast(betas.data()), num_points, num_near_neighbors);
    cudaDeviceSynchronize();

    // compute entropy of current row
    row_sum = ReduceSum(handle, pij, num_near_neighbors, num_points, 0);
    addScalarKernel<<<64,1024>>>(thrust::raw_pointer_cast(row_sum.data()),FLT_MIN,num_points);
cudaDeviceSynchronize();
    thrust::transform(pij.begin(), pij.end(), entropy.begin(),FunctionalEntropy());
    neg_entropy =
        ReduceAlpha(handle, entropy, num_near_neighbors, num_points, -1.0f, 0);

    // binary search for beta
    PerplexitySearchKernel<<<NBLOCKS2, BLOCKSIZE2>>>(
        thrust::raw_pointer_cast(betas.data()),
        thrust::raw_pointer_cast(lower_bound_beta.data()),
        thrust::raw_pointer_cast(upper_bound_beta.data()),
        thrust::raw_pointer_cast(found.data()),
        thrust::raw_pointer_cast(neg_entropy.data()),
        thrust::raw_pointer_cast(row_sum.data()), perplexity_target, epsilon,
        num_points);
    cudaDeviceSynchronize();

    // Check if searching is done
    all_found =
        thrust::reduce(found.begin(), found.end(), 1, thrust::minimum<int>());
    iters++;
  } while (!all_found && iters < 200);
  BroadcastMatrixVector(pij, row_sum, num_near_neighbors, num_points,
                        thrust::divides<float>(), 1, 1.0f);
cudaDeviceSynchronize();
}

template
sparse_matrix<float> perplexityEqualization( int *I, float *D, int n, int nn, float u );
