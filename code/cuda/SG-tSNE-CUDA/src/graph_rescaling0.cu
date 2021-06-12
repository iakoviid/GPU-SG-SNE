/*
    Compute the unnormalized pij matrix given a squared distance matrix and a target perplexity.
    pij = exp(-beta * dist ** 2)

    Note that FAISS returns the first row as the same point, with distance = 0. pii is defined as zero.
*/

#include "kernels/perplexity_search.h"

__global__
void ComputePijKernel(
                      volatile coord * __restrict__ pij,
                      const int* row,
                      const coord * __restrict__ sigma,
                      const unsigned int n,
                      const unsigned int nnz)
{
    register int TID, i, j;
    register coord  beta;

    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= nnz) return;

    i = row;
    s = sigma[i];


    pij[TID] =__expf(-pij[TID] * s);
}

__global__
void PerplexitySearchKernel(
                            volatile coord * __restrict__ sig2,
                            volatile coord * __restrict__ lower_bound,
                            volatile coord * __restrict__ upper_bound,
                            volatile int * __restrict__ found,
                            const coord * __restrict__ neg_entropy,
                            const coord lambda,
                            const coord epsilon,
                            const int num_points)
{
    register int i, is_found;
    register coord perplexity, f_val, sigma, min_sigma, max_sigma;
    i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= num_points) return;

    sigma = sig2[i];

    min_sigma = lower_bound[i];
    max_sigma = upper_bound[i];

    perplexity = neg_entropy[i];
    f_val = perplexity - __logf(lambda);
    is_found = (f_val < epsilon && - f_val < epsilon);
    if (!is_found) {
        if (f_val > 0) {
            min_sigma = sigma;
            sigma = (max_sigma == FLT_MAX || max_sigma == -FLT_MAX) ? sigma * 2.0f : (sigma + max_sigma) / 2.0f;
        } else {
            max_sigma = sigma;
            sigma = (min_sigma == -FLT_MAX || min_sigma == FLT_MAX) ? sigma / 2.0f : (sigma + min_sigma) / 2.0f;
        }
        lower_bound[i] = min_sigma;
        upper_bound[i] = max_sigma;
        sig2[i] = sigma;
    }
    found[i] = is_found;
}
struct FunctionalEntropy {
  __host__ __device__ float operator()(const float &x) const {
      float val = x*log(x);
      return (val != val || isinf(val)) ? 0 : val;
    }
};
void lambdaRescalingGPU(cublasHandle_t &handle,
                      sparse_matrix P,
                      matval lambda)
{   double tolBinary   = 1e-5;
    thrust::device_vector<coord> sig2(P.n, 1.0f);
    thrust::device_vector<coord> lower_bound_sig2(P.n, -FLT_MAX);
    thrust::device_vector<coord> upper_bound_sig2(P.n, FLT_MAX);
    thrust::device_vector<int> found((P.n);

    // TODO: this doesn't really fit with the style
    const int BLOCKSIZE1 = 1024;
    const int NBLOCKS1 = iDivUp(nnz, BLOCKSIZE1);

    const int BLOCKSIZE2 = 128;
    const int NBLOCKS2 = iDivUp((P.n, BLOCKSIZE2);

    size_t iters = 0;
    int all_found = 0;
    thrust::device_vector<coord> neg_entropy;
    do {
      // compute Gaussian Kernel row
      ComputePijKernel<<<NBLOCKS1, BLOCKSIZE1>>>(
                      thrust::raw_pointer_cast(pij.data()),
                      thrust::raw_pointer_cast(squared_dist.data()),
                      thrust::raw_pointer_cast(betas.data()),
                      num_points, num_near_neighbors);
      cudaDeviceSynchronize();
        neg_entropy = ReduceAlpha(handle, entropy, num_near_neighbors, (P.n, -1.0f, 0);

        // binary search for beta
        tsnecuda::PerplexitySearchKernel<<<NBLOCKS2, BLOCKSIZE2>>>(
                                                            thrust::raw_pointer_cast(sig2.data()),
                                                            thrust::raw_pointer_cast(lower_bound_sig2.data()),
                                                            thrust::raw_pointer_cast(upper_bound_sig2.data()),
                                                            thrust::raw_pointer_cast(found.data()),
                                                            thrust::raw_pointer_cast(neg_entropy.data()),
                                                            thrust::raw_pointer_cast(row_sum.data()),
                                                            lambda, tolBinary, (P.n);
        cudaDeviceSynchronize();

        // Check if searching is done
        all_found = thrust::reduce(found.begin(), found.end(), 1, thrust::minimum<int>());
        iters++;
    } while (!all_found && iters < 200);
    // TODO: Warn if iters == 200 because perplexity not found?

    BroadcastMatrixVector(pij, row_sum, num_near_neighbors, (P.n, thrust::divides<coord>(), 1, 1.0f);
}
