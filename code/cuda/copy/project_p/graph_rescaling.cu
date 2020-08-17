/*!
  \file   graph_rescaling.cpp
  \brief  Routines regarding lambda-based graph rescaling.

*/
#include "graph_rescaling.cuh"


/*we will use warps for the parallel evaluation of the expression and the serial code for changing the interval*/
__global__ void bisectionSearchKernel(volatile coord *__restrict__ sig2,
                            double * p_sp,matidx *ir, matidx *jc,int n, double lambda,double tolerance,bool dropLeafEdge) {
        const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
        const unsigned int warp_id = thread_id / 32;
        const unsigned int lane = thread_id % 32;
        unsigned int row = warp_id;
        const unsigned int n_warps=gridDim.x*blockDim.x/32;
        for (;row < n; row+=row+1) {
          int found = 0;
          int iter = 0;
          double sigma = sig2[row];
          double a = -1e3;
          double c = 1e7;
          double sum = 0;
          double perplexity_diff;
          unsigned int row_start = ir[row];
          unsigned int row_end = ir[row + 1];
          while ( __all_sync(FULL_WARP_MASK,  found != 1 && iter < 100 )) {
            sum = 0;
            for (unsigned int element = row_start + lane; element < row_end;
                 element += 1) {
              sum += expf(-p_sp[element] * sigma);
            }
            sum = warp_reduce(sum);
            if (lane == 0) {
              perplexity_diff = sum - lambda;
              if(perplexity_diff<tolerance &&perplexity_diff>-tolerance){found=1;}
              if (perplexity_diff > 0) {
                a = sigma;
                if (c > 1e7) {
                  sigma = 2 * a;
                } else {
                  sigma = 0.5 * (a + c);
                }

              } else {
                c = sigma;
                sigma = 0.5 * (a + c);
              }
              iter++;
            }

          }
          if(lane==0){
          sig2[row]=sigma;
        }
        sum=0;
        for (unsigned int element = row_start + lane; element < row_end;
             element += 1) {
          sum += p_sp[element] ;
        }
        sum = warp_reduce(sum);
        __shfl_sync(FULL_WARP_MASK, sum, 0);
        for (unsigned int element = row_start + lane; element < row_end;
             element += 1) {
          p_sp[element]/=sum;
        }

        // override lambda value of leaf node?
        if ( dropLeafEdge && (row_end-row_start == 1) ) p_sp[row_start] = 0;
        }


        }


void lambdaRescalingGPU( sparse_matrix P, double lambda, bool dist, bool dropLeafEdge ){
  double tolBinary   = 1e-5;
  //int    maxIter     = 100;
  thrust::device_vector<coord> sig2(P.n, 1.0);
  if (dist)  std::cout << "Input considered as distances" << std::endl;
  //if (!dist)  // transform values to distances
  //  for (uint32_t j=P.col[i]; j < P.col[i+1]; j++)
  //    P.val[j] = -log( P.val[j] );

  bisectionSearchKernel<<<1,1>>>(thrust::raw_pointer_cast(sig2.data()),P.val,P.col, P.row,P.n,lambda,tolBinary,dropLeafEdge);
  //CUDA_CALL(cudaDeviceSynchronize());
  //all_found = thrust::reduce(found.begin(), found.end(), 1, thrust::minimum<int>());


}
