/*!
  \file   sgtsne.cpp
  \brief  Entry point to SG-t-SNE

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.


*/

#define CUDART_INF_F __uint2double_rn(0x7ff0000000000000)
#define CUDART_NINF_F __uint2double_rn(0xfff0000000000000)

__global__ normalizeP(P, sum, nnz) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
    P[i]/=sum;
}
}

coord *sgtsne(sparse_matrix P, tsneparams params, coord *y_in,
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
  uint32_t nStoch = makeStochasticGPU(P);
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
  double sum_P = reduce(P.val);

  normalizeP<<<32, 256>>>(P.val, sum_P, P.nnz);

  // ~~~~~~~~~~ extracting BSDB permutation
  idx_t *perm = static_cast<idx_t *>(malloc(P.n * sizeof(idx_t)));
  idx_t *iperm = static_cast<idx_t *>(malloc(P.n * sizeof(idx_t)));

#ifdef FLAG_BSDB_PERM

  std::cout << "Nested dissection permutation..." << std::flush;
  // idx_t options[METIS_NOPTIONS];
  // METIS_SetDefaultOptions(options);
  // options[METIS_OPTION_NUMBERING] = 0;

  int status =
      METIS_NodeND(&P.n, reinterpret_cast<idx_t *>(P.col),
                   reinterpret_cast<idx_t *>(P.row), NULL, NULL, perm, iperm);

  permuteMatrix(&P, perm, iperm);

  if (status != METIS_OK) {
    std::cerr << "METIS error.";
    exit(1);
  }

  std::cout << "DONE" << std::endl;

#else

  for (int i = 0; i < P.n; i++) {
    perm[i] = i;
    iperm[i] = i;
  }

#endif

  // ~~~~~~~~~~ initial embedding coordinates

  coord *y;
  CUDA_CALL(cudaMallocManaged(&y, params.n * params.d * *sizeof(coord)));

  if (y_in == NULL) {

    // ----- Initialize Y
    coord *y_rand =
        static_cast<coord *>(malloc(params.n * params.d * sizeof(coord)));

    for (int i = 0; i < params.n * params.d; i++) {

      y_rand[i] = randn() * .0001;
    }
    CUDA_CALL(cudaMemcpy(y, y_rand, params.n * params.d * *sizeof(coord),
                         cudaMemcpyHostToDevice));

  } else {
    CUDA_CALL(cudaMemcpy(y, y_in, params.n * params.d * *sizeof(coord),
                         cudaMemcpyHostToDevice));
  }

  // ~~~~~~~~~~ gradient descent
  kl_minimization(y, params, csb, timeInfo);

  // ~~~~~~~~~~ inverse permutation
  coord *y_inv;
  CUDA_CALL(cudaMallocManaged(&y_inv, params.n * params.d * *sizeof(coord)));
  ApplyPermutation(y_inv, y, params.n, params.d, iperm);
  coord *y_return =
      static_cast<coord *>(malloc(params.n * params.d * sizeof(coord)));
  CUDA_CALL(cudaMemcpy(y_return, y, params.n * params.d * *sizeof(coord),
                       cudaMemcpyDeviceToHost));

  // ~~~~~~~~~~ dellocate memory

  deallocate(csb);
  cudaFree(y);
  cudaFree(perm);
  cudaFree(iperm);

  return y_return;
}

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
__global__ void ComputePijKernel(double *val_P, double *distances,
                               double perplexity, int nn) {
  const unsigned int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned int warp_id = thread_id / 32;
  const unsigned int lane = thread_id % 32;
  const unsigned int row = warp_id;
  const unsigned int n_warps = gridDim.x * blockDim.x / 32;

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
          while ( __all_sync(FULL_WARP_MASK,  found != 1 && iter < 200 ) {
      sum = 0;
      H = .0;
      for (unsigned int element = lane; element < nn; element += 32) {
        val_P[element + row * nn] = exp(-beta * distances[element + 1 + row * (nn + 1)]);
        sum += val_P[element + row * nn];
        H += beta * distances[element + 1 + row * (nn + 1)] * val_P[element + row * nn];
      }
      sum = warp_reduce(sum);
      H = warp_reduce(H);

      if (lane == 0) {
        H = (H / sum) + log(sum);
        double Hdiff = H - log(perplexity);
        if (Hdiff < tol && -Hdiff < tol) {
          found = 1;
        }

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
            beta = (beta + max_beta) / 2.0;
        }
        iter++;
      }
      __shfl_sync(FULL_WARP_MASK, beta, 0);


          }


        __shfl_sync(FULL_WARP_MASK, sum, 0);
        for (unsigned int element =  lane; element < nn;
             element += 32) {
      val_P[element + row * nn] /= sum;
            }
            if(lane==0){ic[row]=row*nn;}
            for (unsigned int element = lane; element < nn; element += 32) {
              ir[row*nn+element]=I[row*(nn+1)+element+1];
            }

  }
  if (thread_id==0){if(row==n){col[n]=n*nn;}

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

  ComputePijKernel<<<32, 256>>>(&val[i * nn], &D[i * (nn + 1)], u, nn);



  if (nz != (nn * n))
    std::cerr << "Problem with kNN graph..." << std::endl;

  P.n = n;
  P.m = n;
  P.nnz = n * nn;
  P.row = row;
  P.col = col;
  P.val = val;

  return P;
}
