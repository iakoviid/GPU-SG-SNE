#include "Frep.hpp"
#include "gradient_descend.cuh"
#include <math.h>

__global__ void compute_dy(coord *dy, coord *Fattr, coord *Frep, int n, int d,
                           coord alpha) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n * d;
       TID += gridDim.x * blockDim.x) {
    dy[TID] = (alpha * Fattr[TID]) - Frep[TID];
  }
}
__global__ void gradient_update(coord *dY, coord *uY, int N, int no_dims,
                                coord *Y, coord *gains, coord momentum,
                                coord eta) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N * no_dims;
       i += gridDim.x * blockDim.x) {
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    if (gains[i] < .01)
      gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];
  }
}

void update_positions(coord *dY, coord *uY, int n, int d, coord *Y,
                      coord *gains, coord momentum, coord eta) {

  gradient_update<<<32, 256>>>(dY, uY, n, d, Y, gains, momentum, eta);
  coord *meany = (coord *)malloc(d * sizeof(coord));

  thrust::device_ptr<double> yVec_ptr = thrust::device_pointer_cast(Y);

  for (int i = 0; i < d; i++) {
    meany[i] = thrust::reduce(yVec_ptr + (i)*n, yVec_ptr + (i + 1) * n) / n;

    addScalar<<<32, 256>>>(&Y[i * n], -meany[i], n);
  }
}


template <class dataPoint>
double compute_gradient(dataPoint *dy, double *timeFrep, double *timeFattr,
                        tsneparams params, dataPoint *y, sparse_matrix P) {
  double timeInfo[7];
  // ----- parse input parameters
  int d = params.d;
  int n = params.n;
  // ----- timing
  struct GpuTimer timer;

  // ----- Allocate memory

  dataPoint *Fattr;
  dataPoint *Frep;
  CUDA_CALL(cudaMallocManaged(&Fattr, n * d * sizeof(dataPoint)));
  CUDA_CALL(cudaMallocManaged(&Frep, n * d * sizeof(dataPoint)));
  initKernel<<<64, 1024>>>(Fattr, 0.0, n * d);
  initKernel<<<64, 1024>>>(Frep, 0.0, n * d);

  // ------ Compute PQ (fattr)
  timer.Start();
  AttractiveEstimation(P.row, P.col, P.val, Fattr, y, n, d, 0, 0, 0, P.nnz,
                       P.format);
  timer.Stop();
  *timeFattr = timer.Elapsed();

  // ------ Compute QQ (frep)
  timer.Start();
  double zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, timeInfo);
  timer.Stop();
  *timeFrep = timer.Elapsed();

  // ----- Compute gradient (dY)
  compute_dy<<<32, 256>>>(dy, Fattr, Frep, n, d, params.alpha);

  // ----- Free-up memory
  CUDA_CALL(cudaFree(Fattr));
  CUDA_CALL(cudaFree(Frep));

  return zeta;
}
void kl_minimization(coord *y, tsneparams params, sparse_matrix P) {
  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;
  // int iterPrint = 50;

  double timeFattr = 0.0;
  double timeFrep = 0.0;

  int n = params.n;
  int d = params.d;
  int max_iter = params.maxIter;

  coord zeta = 0;

  // ----- Allocate memory
  coord *dy;
  coord *uy;
  coord *gains;
  CUDA_CALL(cudaMallocManaged(&dy, d * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&uy, d * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&gains, d * n * sizeof(coord)));

  /*-------Initialize-----*/
  initKernel<<<64, 1024>>>(uy, 0.0, n * d);
  initKernel<<<64, 1024>>>(gains, 1.0, n * d);

  for (int iter = 0; iter < max_iter; iter++) {

    printf("%d--------------------------------------\n", iter);

    zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, P);

    update_positions(dy, uy, n, d, y, gains, momentum, eta);

    // Stop lying about the P-values after a while, and switch momentum
    if (iter == stop_lying_iter) {
      params.alpha = 1;
    }

    // Change momentum after a while
    if (iter == mom_switch_iter) {
      momentum = final_momentum;
    }
  }

  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(uy));
  CUDA_CALL(cudaFree(gains));
}
