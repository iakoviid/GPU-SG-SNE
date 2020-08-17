#include "gradient_descend.cuh"



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
    printf("cuda mean %lf \n", meany[i]);
    addScalar<<<32, 256>>>(&Y[i * n], -meany[i], n);
  }
}

template <class dataPoint>
double compute_gradient(dataPoint *dy, double *timeFrep, double *timeFattr,
                        tsneparams params, dataPoint *y) {
  // ----- parse input parameters
  int d = params.d;
  int n = params.n;

  // ----- timing
  //struct timeval start;

  // ----- Allocate memory

  coord* Fattr;
  coord* Frep;
  CUDA_CALL(cudaMallocManaged(&Fattr,n*d*sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&Frep,n*d*sizeof(coord)));
  // ------ Compute PQ (fattr)
  // start = tsne_start_timer();
  // csb_pq( NULL, NULL, csb, y, Fattr, n, d, 0, 0, 0 );
  double zeta=0;
  zeta = computeFrepulsive_interp(Frep, y, n, d, params.h);

  /*
if (timeInfo != nullptr)
  zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np,
                                       &timeInfo[1]);
else
zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np);
  */
  //*timeFrep += tsne_stop_timer("QQ", start);
  // double zeta = computeFrepulsive_exact(Frep, y, n, d);

  // ----- Compute gradient (dY)
  compute_dy<<<32,256>>>(dy, Fattr, Frep, n, d, params.alpha);

  // ----- Free-up memory
  cudaFree(Fattr);
  cudaFree(Frep);
  return zeta;
}
void kl_minimization(coord *y, tsneparams params) {
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
/*
  // ----- Print precision
  if (sizeof(y[0]) == 4)
    std::cout << "Working with single precision" << std::endl;
  else if (sizeof(y[0]) == 8)
    std::cout << "Working with double precision" << std::endl;
  */// ----- Start t-SNE iterations
  printf("CUDA~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n");
  for (int iter = 0; iter < max_iter; iter++) {

    if(iter%10==0){printf("Cuda Iteration=%d\n",iter );}

    zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y);

    coord* y_h=(coord *) malloc(n*d*sizeof(coord));
    cudaMemcpy(y_h, dy, d * n * sizeof(coord), cudaMemcpyDeviceToHost);

    printf("Cuda zeta=%lf\n",zeta );
    /*
    for(int i=0;i<n;i++){
      for(int j=0;j<d;j++){
        //printf("dy=%lf\n",y_h[i+j*n] );
      }
    }*/
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
/*
  // ----- Print statistics (time spent at PQ and QQ)
  std::cout << " --- Time spent in each module --- \n" << std::endl;
  std::cout << " Attractive forces: " << timeFattr << " sec ["
            << timeFattr / (timeFattr + timeFrep) * 100
            << "%] |  Repulsive forces: " << timeFrep << " sec ["
            << timeFrep / (timeFattr + timeFrep) * 100 << "%]" << std::endl;
 */
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(uy));
  CUDA_CALL(cudaFree(gains));
}
