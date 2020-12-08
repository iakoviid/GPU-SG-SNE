/*!
  \file   sgtsne.cpp
  \brief  Entry point to SG-t-SNE

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.


*/
#include "sgtsne.cuh"
#define CUDART_INF_F __uint2double_rn(0x7ff0000000000000)
#define CUDART_NINF_F __uint2double_rn(0xfff0000000000000)
#include "graph_rescaling.hpp"

#include "sparse_reorder.cuh"

//#define FLAG_BSDB_PERM

coord *sgtsneCUDA(sparse_matrix P, tsneparams params, coord *y_in,
                  double **timeInfo) {
  // ~~~~~~~~~~ unless h is specified, use default ones
  if (params.h <= 0)
    switch (params.d) {
    case 1:
      params.h = 0.5;
      break;
    case 2:
      params.h = 0.7;
      break;
    case 3:
      params.h = 1.2;
      break;
    }

  // ~~~~~~~~~~ print input parameters
  printParams(params);

  // ~~~~~~~~~~ make sure input matrix is column stochastic
  uint32_t nStoch = makeStochastic(P);
  std::cout << nStoch << " out of " << P.n << " nodes already stochastic"
            << std::endl;

  // ~~~~~~~~~~ prepare graph for SG-t-SNE

  // ----- lambda rescaling
  if (params.lambda == 1)
    std::cout << "Skipping λ rescaling..." << std::endl;
  else
    lambdaRescaling(P, params.lambda, false, params.dropLeaf);

  // ----- symmetrizing
  symmetrizeMatrix(&P);

  // ----- normalize matrix (total sum is 1.0)
  double sum_P = .0;
  for (int i = 0; i < P.nnz; i++) {
    sum_P += P.val[i];
  }
  for (int i = 0; i < P.nnz; i++) {
    P.val[i] /= sum_P;
  }

  // ~~~~~~~~~~ extracting BSDB permutation
  int *perm = static_cast<int *>(malloc(P.n * sizeof(int)));
  sparse_matrix Pd;
  Pd.n=P.n;
  Pd.nnz=P.nnz;
  Pd.m=P.m;
  CUDA_CALL(cudaMallocManaged(&Pd.row, (P.n+1) * sizeof(int)));
  CUDA_CALL(cudaMallocManaged(&Pd.col, P.nnz * sizeof(int)));
  CUDA_CALL(cudaMallocManaged(&Pd.val, P.nnz * sizeof(coord)));

#ifdef FLAG_BSDB_PERM
  double *csr_val_permuted = (double *)calloc(P.nnz, sizeof(double));
  int *csr_col_permuted = (int *)calloc(P.nnz, sizeof(int));
  int *csr_row_permuted = (int *)calloc(P.n + 1, sizeof(int));
  cusolverSpHandle_t handle = NULL;
  cusparseHandle_t cusparseHandle = NULL; /* used in residual evaluation */
  cudaStream_t stream = NULL;
  cusparseMatDescr_t descrA = NULL;
  checkCudaErrors(cusolverSpCreate(&handle));
  checkCudaErrors(cusparseCreate(&cusparseHandle));

  checkCudaErrors(cudaStreamCreate(&stream));
  /* bind stream to cusparse and cusolver*/
  checkCudaErrors(cusolverSpSetStream(handle, stream));
  checkCudaErrors(cusparseSetStream(cusparseHandle, stream));

  /* configure matrix descriptor*/
  checkCudaErrors(cusparseCreateMatDescr(&descrA));
  checkCudaErrors(cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
  int baseA = 0; /* base index in CSR format */
  if (baseA) {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ONE));
  } else {
    checkCudaErrors(cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
  }

  SparseReorder("metis", handle, descrA, P.m, P.n, P.nnz, P.col, P.row, P.val,
                csr_row_permuted, csr_col_permuted, csr_val_permuted, perm);
  // Transfer P to GPU memory prepare format
  CUDA_CALL(cudaMemcpy(Pd.row, csr_row_permuted, (P.n + 1) * sizeof(int),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Pd.col, csr_col_permuted, P.nnz * sizeof(int),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Pd.val, csr_val_permuted, P.nnz * sizeof(coord),
                       cudaMemcpyHostToDevice));

  free(csr_val_permuted);
  free(csr_col_permuted);
  free(csr_row_permuted);
  if (handle) {
    checkCudaErrors(cusolverSpDestroy(handle));
  }
  if (cusparseHandle) {
    checkCudaErrors(cusparseDestroy(cusparseHandle));
  }
  if (stream) {
    checkCudaErrors(cudaStreamDestroy(stream));
  }
  if (descrA) {
    checkCudaErrors(cusparseDestroyMatDescr(descrA));
  }

#else
  // Transfer P to GPU memory prepare format
  CUDA_CALL(cudaMemcpy(Pd.row, P.col, (P.n + 1) * sizeof(matidx),cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Pd.col, P.row, P.nnz * sizeof(matidx), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMemcpy(Pd.val, P.val, P.nnz * sizeof(coord), cudaMemcpyHostToDevice));
  for (int i = 0; i < P.n; i++) {
    perm[i] = i;
  }

#endif

  // ~~~~~~~~~~ initial embedding coordinates

  coord *y;
  CUDA_CALL(cudaMallocManaged(&y, params.n * params.d * sizeof(coord)));

  if (y_in == NULL) {

    // ----- Initialize Y
    coord *y_rand =
        static_cast<coord *>(malloc(params.n * params.d * sizeof(coord)));

    for (int i = 0; i < params.n * params.d; i++) {

      y_rand[i] = randn() * .0001;
    }
    CUDA_CALL(cudaMemcpy(y, y_rand, params.n * params.d * sizeof(coord),
                         cudaMemcpyHostToDevice));
    cudaFree(y_rand);

  } else {
    CUDA_CALL(cudaMemcpy(y, y_in, params.n * params.d * sizeof(coord),
                         cudaMemcpyHostToDevice));
  }


  // ~~~~~~~~~~ gradient descent
  kl_minimization(y, params, Pd);

  // ~~~~~~~~~~ inverse permutation

  coord *y_copy =
      static_cast<coord *>(malloc(params.n * params.d * sizeof(coord)));

  CUDA_CALL(cudaMemcpy(y_copy, y, params.n * params.d * sizeof(coord),
                       cudaMemcpyDeviceToHost));

  coord *y_return =
      static_cast<coord *>(malloc(params.n * params.d * sizeof(coord)));
  for (int i = 0; i < params.n; i++) {
    for (int j = 0; j < params.d; j++) {
      y_return[perm[i] * params.d + j] = y_copy[i * params.d + j];
    }
  }
  // ~~~~~~~~~~ dellocate memory
  cudaFree(y);
  free(y_copy);
  free(perm);
  return y_return;
}
/*
__global__ void ApplyPermutation(coord *y_perm, coord *y, int *perm) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < n;
       i += gridDim.x * blockDim.x) {
    for (int j = 0; j < d; j++) {
      y_perm[i + n * j] = y[perm[i] + j]
    }
  }
}

/*we will use warps for the parallel evaluation of the expression and the serial
 * code for changing the interval*/
/*warps fit as the graph is k-regular*/
/*
__global__ void ComputePijKernel(double *val_P, double *distances,
                                 double perplexity, int nn, matidx *ir,
                                 matidx *ic) {
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int lane = thread_id % 32;
  const unsigned int row = warp_id;
  const unsigned int n_warps = gridDim.x * blockDim.x / 32;
  const unsigned int warp_id_block = threadIdx.x / 32;
  __shared__ coord sdata[Blocksize / 32];

  if (; row < n; row += row + n_warps) {
    int found = 0;
    int iter = 0;
    double beta = 1;
    double min_beta = CUDART_NINF_F;
    double max_beta = CUDART_INF_F;
    double tol = 1e-5;
    double sum = 0;
    double H = .0;

    // Iterate until we found a good perplexity
    while (__all_sync(FULL_WARP_MASK, found != 1) && iter < 200) {
      sum = 0;
      H = .0;
      for (unsigned int element = lane; element < nn; element += 32) {
        val_P[element + row * nn] =
            exp(-beta * distances[element + 1 + row * (nn + 1)]);
        sum += val_P[element + row * nn];
        H += beta * distances[element + 1 + row * (nn + 1)] *
             val_P[element + row * nn];
      }
      sum = warp_reduce(sum);
      H = warp_reduce(H);

      if (lane == 0) {
        H = (H / sum) + log(sum);
        double Hdiff = H - log(perplexity);
        if (Hdiff < tol && -Hdiff < tol) {
          found = 1;
        } else {

          if (Hdiff > 0) {
            min_beta = beta;
            if (max_beta == CUDART_INF_F || max_beta == -CUDART_NINF_F)
              beta *= 2.0;
            else
              beta = (beta + max_beta) / 2.0;

          } else {
            max_beta = beta;
            if (min_beta == CUDART_INF_F || min_beta == -CUDART_NINF_F)
              beta /= 2.0;
            else
              beta = (beta + min_beta) / 2.0;
          }
        }
        sdata[warp_id_block] = beta;
      }
      iter++;
      __syncwarp(FULL_WARP_MASK);
      beta = sdata[warp_id_block];
    }
    if (lane == 0) {
      sdata[warp_id_block] = sum;
    }
    __syncwarp(FULL_WARP_MASK);
    sum = sdata[warp_id_block];
    for (unsigned int element = lane; element < nn; element += 32) {
      val_P[element + row * nn] /= sum;
    }

    if (lane == 0) {
      ir[row] = row * nn;
    }
    for (unsigned int element = lane; element < nn; element += 32) {
      ic[row * nn + element] = I[row * (nn + 1) + element + 1];
    }
  }
  if (thread_id == 0) {
    {
      ir[n] = n * nn;
    }
  }

  sparse_matrix perplexityEqualizationGPU(int *I, double *D, int n, int nn,
                                          double u) {

    sparse_matrix P;
    matval *val;
    matidx *row, *col;

    // allocate space for CSC format
    CUDA_CALL(cudaMallocManaged(&val, (nn)*n * sizeof(coord)));
    CUDA_CALL(cudaMallocManaged(&row, (nn)*n * sizeof(matidx)));
    CUDA_CALL(cudaMallocManaged(&val, (n + 1) * sizeof(matidx)));

    // perplexity-equalization of kNN input

    ComputePijKernel<<<32, 256>>>(val, D, u, nn, row, col);

    P.n = n;
    P.m = n;
    P.nnz = n * nn;
    P.row = row;
    P.col = col;
    P.val = val;

    return P;
  }*/
