#include "compute_error.cu"
#include "gradient_descend.cuh"
#include <fstream>
#include <math.h>
cudaStream_t streamAttr=0;
cudaStream_t streamRep=0;
int grid[1000];
int iteration=0;
using namespace std;
template <class dataPoint>
__global__ void compute_dy(dataPoint *dy, dataPoint *Fattr, dataPoint *Frep,
                           int n, int d, dataPoint alpha) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n * d;
       TID += gridDim.x * blockDim.x) {
    dy[TID] = (alpha * Fattr[TID]) - Frep[TID];
  }
}

template <class dataPoint>
__device__ __host__ static inline dataPoint sign(dataPoint x) {

  return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
}

template <class dataPoint>
__global__ void gradient_update(dataPoint *dY, dataPoint *uY, int N,
                                int no_dims, dataPoint *Y, dataPoint *gains,
                                dataPoint momentum, dataPoint eta) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < N * no_dims;
       i += gridDim.x * blockDim.x) {
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    if (gains[i] < .01)
      gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];
  }
}

template <class dataPoint>
void update_positions(dataPoint *dY, dataPoint *uY, int n, int d, dataPoint* Y,
                      dataPoint *gains, dataPoint momentum, dataPoint eta) {

  gradient_update<<<64, 1024>>>(dY, uY, n, d, Y, gains, momentum, eta);
/*
  dataPoint miny[4];

    for (int j = 0; j < d; j++) {
      miny[j] = thrust::reduce(thrust::cuda::par.on(streamRep), Y.begin() + j * n,
                               Y.begin() + n * (j + 1), 100.0,
                               thrust::minimum<dataPoint>());
    }

    for (int j = 0; j < d; j++) {
    addScalar<<<64, 1024, 0, streamRep>>>(thrust::raw_pointer_cast(Y.data())+j*n, -miny[j],n);
    }
*/
thrust::device_ptr<dataPoint> yVec_ptr(Y);
dataPoint meany[4];
  for (int i = 0; i < d; i++) {
    meany[i] = thrust::reduce(yVec_ptr + (i)*n, yVec_ptr + (i + 1) * n) / n;
    addScalar<<<32, 256>>>(Y+i*n, -meany[i], n);
  }

}

float compute_gradient(float *dy, double *timeFrep, double *timeFattr,
                           tsneparams params, float *y,
                           sparse_matrix<float> P, double *timeInfo,
                           float *errorRep = nullptr) {
  //double timeInfo[7] = {0};
  // ----- parse input parameters
  int d = params.d;
  int n = params.n;
  float h =params.h;
  int num_threads= 1024;

  const int num_blocks = (n + num_threads - 1) / num_threads;

  // ----- timing
  struct GpuTimer timer;

  // ----- Allocate memory
  thrust::device_ptr<float> yVec_ptr(y);
  float miny[4];
  for (int j = 0; j < d; j++) {
    miny[j] = thrust::reduce(yVec_ptr+j*n, yVec_ptr+n*(j+1),100.0, thrust::minimum<float>());
    addScalar<<<num_blocks, num_threads>>>(&y[j * n], -miny[j], n);
  }
  float maxy = thrust::reduce(yVec_ptr, yVec_ptr + n * d, 0.0,
                                thrust::maximum<float>());

    int nGrid = std::max((int)std::ceil(maxy / h), 14);
    nGrid = getBestGridSize(nGrid);
  grid[iteration]=nGrid+2;
  iteration++;
   cufftHandle plan, plan_rhs;
    int n2 = nGrid + 2;
    switch (d) {
    case 1: {
      int ng[1] = {(int)n2};
      cufftPlan1d(&plan, n2, CUFFT_C2C, 1);
      cufftPlanMany(&plan_rhs, 1, ng, NULL, 1, n2, NULL, 1, n2, CUFFT_C2C, d + 1);
      break;
    }
    case 2: {
      int ng[2] = {(int)n2, (int)n2};
      cufftPlanMany(&plan, 2, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 1);
      cufftPlanMany(&plan_rhs, 2, ng, NULL, 1, n2 * n2, NULL, 1, n2 * n2,
                    CUFFT_C2C, d + 1);
      break;
    }
    case 3: {
      int ng[3] = {(int)n2, (int)n2, (int)n2};
      cufftPlanMany(&plan, 3, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 1);
      cufftPlanMany(&plan_rhs, 3, ng, NULL, 1, n2 * n2 * n2, NULL, 1,
                    n2 * n2 * n2, CUFFT_C2C, d + 1);
      break;
    }
    }
    //cufftSetStream(plan, streamRep);
    //cufftSetStream(plan_rhs, streamRep);
    int m=d+1;
    int nVec=d+1;
    /*Allocate memory*/
    float *yt;
    CUDA_CALL(cudaMallocManaged(&yt, (d) * n * sizeof(float)));
    float *VScat;
    CUDA_CALL(cudaMallocManaged(&VScat, (d+1) * n * sizeof(float)));
    float *PhiScat;
    CUDA_CALL(cudaMallocManaged(&PhiScat, (d+1) * n * sizeof(float)));
    int szV = pow(nGrid + 2, d) * m;
    float *VGrid;
    CUDA_CALL(cudaMallocManaged(&VGrid, szV * sizeof(float)));
    float *PhiGrid;
    CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(float)));
    Complex *Kc, *Xc;
    CUDA_CALL(cudaMallocManaged(&Kc, szV * sizeof(Complex)));
    CUDA_CALL(cudaMallocManaged(&Xc, nVec *szV * sizeof(Complex)));
    thrust::device_vector<float> zetaVec(n);

  float *Fattr;
  float *Frep;
  CUDA_CALL(cudaMallocManaged(&Fattr, n * d * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&Frep, n * d * sizeof(float)));
  initKernel<<<64, 1024>>>(Fattr, (float)0.0, n * d);
  initKernel<<<64, 1024>>>(Frep, (float)0.0, n * d);

  // ------ Compute PQ (fattr)
  timer.Start();
  AttractiveEstimation(P, d, y, Fattr);
  timer.Stop();
  *timeFattr = timer.Elapsed();
  timeInfo[0]=*timeFattr;
  // ------ Compute QQ (frep)
  timer.Start();
  float zeta =
      computeFrepulsive_interp(Frep, y, n, d, (float)h, timeInfo,nGrid, plan, plan_rhs,yt,VScat,PhiScat,VGrid,PhiGrid,Kc,Xc,zetaVec);
  timer.Stop();
  *timeFrep += timer.Elapsed();


  //for (int i = 0; i < 7; i++) {TotaltimeInfo[i] += timeInfo[i];}

  // ----- Compute gradient (dY)

  cudaDeviceSynchronize();
  compute_dy<<<64, 1024>>>(dy, Fattr, Frep, n, d, (float)params.alpha);

    if (params.ComputeError > 0 && errorRep != nullptr) {
        *errorRep = computeErrorCPUrmse(Frep, y, n, d);
    }

  // ----- Free-up memory
  cufftDestroy(plan);
  cufftDestroy(plan_rhs);
  CUDA_CALL(cudaFree(PhiGrid));
  CUDA_CALL(cudaFree(VGrid));
  CUDA_CALL(cudaFree(yt));
  CUDA_CALL(cudaFree(VScat));
  CUDA_CALL(cudaFree(PhiScat));
  CUDA_CALL(cudaFree(Kc));
  CUDA_CALL(cudaFree(Xc));
  CUDA_CALL(cudaFree(Fattr));
  CUDA_CALL(cudaFree(Frep));

  return zeta;
}

double compute_gradient(double *dy, double *timeFrep, double *timeFattr,
                           tsneparams params, double *y,
                           sparse_matrix<double> P, double *timeInfo,
                           double *errorRep = nullptr) {
  //double timeInfo[7] = {0};
  // ----- parse input parameters
  int d = params.d;
  int n = params.n;
  double h =params.h;
  int num_threads= 1024;

  const int num_blocks = (n + num_threads - 1) / num_threads;

  // ----- timing
  struct GpuTimer timer;

  // ----- Allocate memory
  thrust::device_ptr<double> yVec_ptr(y);
  double miny[4];
  for (int j = 0; j < d; j++) {
    miny[j] = thrust::reduce(yVec_ptr+j*n, yVec_ptr+n*(j+1),100.0, thrust::minimum<double>());
    addScalar<<<num_blocks, num_threads>>>(&y[j * n], -miny[j], n);
  }
  double maxy = thrust::reduce(yVec_ptr, yVec_ptr + n * d, 0.0,
                                thrust::maximum<double>());

    int nGrid = std::max((int)std::ceil(maxy / h), 14);
    nGrid = getBestGridSize(nGrid);
    cufftHandle plan, plan_rhs;
    int n2 = nGrid + 2;
    switch (d) {
    case 1: {
      int ng[1] = {(int)n2};
      cufftPlan1d(&plan, n2, CUFFT_Z2Z, 1);
      cufftPlanMany(&plan_rhs, 1, ng, NULL, 1, n2, NULL, 1, n2, CUFFT_Z2Z, d+1);
      break;
    }
    case 2: {
      int ng[2] = {(int)n2, (int)n2};
      cufftPlanMany(&plan, 2, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
      cufftPlanMany(&plan_rhs, 2, ng, NULL, 1, n2 * n2, NULL, 1, n2 * n2, CUFFT_Z2Z,
                    d+1);
      break;
    }
    case 3: {
      int ng[3] = {(int)n2, (int)n2, (int)n2};
      cufftPlanMany(&plan, 3, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
      cufftPlanMany(&plan_rhs, 3, ng, NULL, 1, n2 * n2 * n2, NULL, 1, n2 * n2 * n2,
                    CUFFT_Z2Z, d+1);
      break;
    }
    }
    //cufftSetStream(plan, streamRep);
    //cufftSetStream(plan_rhs, streamRep);
    int m=d+1;
    int nVec=d+1;
    /*Allocate memory*/
    double *yt;
    CUDA_CALL(cudaMallocManaged(&yt, (d) * n * sizeof(double)));
    double *VScat;
    CUDA_CALL(cudaMallocManaged(&VScat, (d+1) * n * sizeof(double)));
    double *PhiScat;
    CUDA_CALL(cudaMallocManaged(&PhiScat, (d+1) * n * sizeof(double)));
    int szV = pow(nGrid + 2, d) * m;
    double *VGrid;
    CUDA_CALL(cudaMallocManaged(&VGrid, szV * sizeof(double)));
    double *PhiGrid;
    CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(double)));
    ComplexD *Kc, *Xc;
    CUDA_CALL(cudaMallocManaged(&Kc, szV * sizeof(ComplexD)));
    CUDA_CALL(cudaMallocManaged(&Xc, nVec *szV * sizeof(ComplexD)));
    thrust::device_vector<double> zetaVec(n);

  double *Fattr;
  double *Frep;
  CUDA_CALL(cudaMallocManaged(&Fattr, n * d * sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&Frep, n * d * sizeof(double)));
  initKernel<<<64, 1024>>>(Fattr, (double)0.0, n * d);
  initKernel<<<64, 1024>>>(Frep, (double)0.0, n * d);

  // ------ Compute PQ (fattr)
  timer.Start();
  AttractiveEstimation(P, d, y, Fattr);
  timer.Stop();
  *timeFattr += timer.Elapsed();

  // ------ Compute QQ (frep)
  timer.Start();
  double zeta =
      computeFrepulsive_interp(Frep, y, n, d, (double)h, timeInfo,nGrid, plan, plan_rhs,yt,VScat,PhiScat,VGrid,PhiGrid,Kc,Xc,zetaVec);
  timer.Stop();
  *timeFrep += timer.Elapsed();


  //for (int i = 0; i < 7; i++) {TotaltimeInfo[i] += timeInfo[i];}

  // ----- Compute gradient (dY)

  cudaDeviceSynchronize();
  compute_dy<<<64, 1024>>>(dy, Fattr, Frep, n, d, (double)params.alpha);

    if (params.ComputeError > 0 && errorRep != nullptr) {
        *errorRep = computeErrorCPUrmse(Frep, y, n, d);
    }

  // ----- Free-up memory
  cufftDestroy(plan);
  cufftDestroy(plan_rhs);
  CUDA_CALL(cudaFree(PhiGrid));
  CUDA_CALL(cudaFree(VGrid));
  CUDA_CALL(cudaFree(yt));
  CUDA_CALL(cudaFree(VScat));
  CUDA_CALL(cudaFree(PhiScat));
  CUDA_CALL(cudaFree(Kc));
  CUDA_CALL(cudaFree(Xc));
  CUDA_CALL(cudaFree(Fattr));
  CUDA_CALL(cudaFree(Frep));

  return zeta;
}
void kl_minimization(float *y, tsneparams params,
                     sparse_matrix<float> P,double* timeInfo) {


  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  float momentum = .5, final_momentum = .8;
  float eta = 200.0;
    //std::cout << "scientific:\n" << std::scientific;
  // int iterPrint = 50;
  //cudaStreamCreate(&streamAttr);
  //cudaStreamCreate(&streamRep);
  double timeFattr = 0.0;
  double timeFrep = 0.0;

  int n = params.n;
  int d = params.d;
  int max_iter = params.maxIter;

  float zeta = 0;
  float *errorRep = nullptr;
  int errorCalcs = (int)max_iter / 10;
  if (params.ComputeError > 0) {
    errorRep = (float *)malloc(sizeof(float) * errorCalcs);
  }
  // ----- Allocate memory
  float *dy;
  float *uy;
  float *gains;
  CUDA_CALL(cudaMallocManaged(&dy, d * n * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&uy, d * n * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&gains, d * n * sizeof(float)));

  /*-------Initialize-----*/
  initKernel<<<64, 1024>>>(uy, (float)0.0, n * d);
  initKernel<<<64, 1024>>>(gains, (float)1.0, n * d);
    struct timeval t1, t2;
  gettimeofday(&t1, NULL);

  for (int iter = 0; iter < max_iter; iter++) {

    if (iter % 100 == 0) {
      printf("---------------------------%d---------------------------\n",
             iter);
    }

    if (params.ComputeError > 0 && iter % 10 == 0) {
      zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, P, &timeInfo[iter*7],
                              &errorRep[(int)iter / 10]);
    } else {
      zeta =
          compute_gradient(dy, &timeFrep, &timeFattr, params, y, P, &timeInfo[iter*7]);
    }

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
  gettimeofday(&t2, NULL);
  double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("time=%lf\n",elapsedTime );
  timeInfo[7*1000]=elapsedTime;

   ofstream ngrid;
   ngrid.open("Grid.txt");
    for(int i=0;i<1000;i++){
	ngrid<<grid[i]<<"\n";
	}
    ngrid.close();

  if (params.ComputeError > 0) {
    ofstream errorf;
    errorf.open("errorInfo.txt");
    for (int i = 0; i < errorCalcs; i++) {
      errorf << errorRep[i] << "\n";
    }
    errorf.close();
    free(errorRep);
  }

  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(uy));
  CUDA_CALL(cudaFree(gains));
}
void kl_minimization(double *y, tsneparams params,
                     sparse_matrix<double> P) {

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;
    std::cout << "scientific:\n" << std::scientific;
  // int iterPrint = 50;
  //cudaStreamCreate(&streamAttr);
  //cudaStreamCreate(&streamRep);
  double timeFattr = 0.0;
  double timeFrep = 0.0;

  int n = params.n;
  int d = params.d;
  int max_iter = params.maxIter;

  double zeta = 0;
  double *errorRep = nullptr;
  int errorCalcs = (int)max_iter / 10;
  if (params.ComputeError > 0) {
    errorRep = (double *)malloc(sizeof(double) * errorCalcs);
  }
  // ----- Allocate memory
  double *dy;
  double *uy;
  double *gains;
  CUDA_CALL(cudaMallocManaged(&dy, d * n * sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&uy, d * n * sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&gains, d * n * sizeof(double)));

  /*-------Initialize-----*/
  initKernel<<<64, 1024>>>(uy, (double)0.0, n * d);
  initKernel<<<64, 1024>>>(gains, (double)1.0, n * d);
  //double timeInfo[7] = {0};
   double timeInfo[1000*7]={0};
  for (int iter = 0; iter < max_iter; iter++) {

    if (iter % 100 == 0) {
      printf("---------------------------%d---------------------------\n",
             iter);
    }

    if (params.ComputeError > 0 && iter % 10 == 0) {
      zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, P, &timeInfo[iter*7],
                              &errorRep[(int)iter / 10]);
    } else {
      zeta =
          compute_gradient(dy, &timeFrep, &timeFattr, params, y, P, &timeInfo[iter*7]);
    }

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
  cout << "Time for computing Attractive Estimation " << timeFattr << "ms\n";
  cout << "Time for computing Repulsive Estimation " << timeFrep << " ms\n";

  cout << "Detailed " << timeInfo[1] << " ms in s2g, " << timeInfo[2]
       << " ms in g2g, " << timeInfo[3] << " ms in g2s\n";
  cout << timeInfo[4] << " ms in zetaAndForce, " << timeInfo[5]
       << " ms  in nuconv, " << timeInfo[6] << " ms in preprocessing\n";
  cout << "and " << timeInfo[0] << " in permutations\n";

  ofstream myfile;
  myfile.open("timeInfo.txt");
  for(int i=0;i<max_iter;i++){
    for(int j=0;j<7;j++){
      myfile<<timeInfo[7*i+j]<<" ";
    }
    myfile<<"\n";
  }  myfile.close();


  if (params.ComputeError > 0) {
    ofstream errorf;
    errorf.open("errorInfo.txt");
    for (int i = 0; i < errorCalcs; i++) {
      errorf << errorRep[i] << "\n";
    }
    errorf.close();
    free(errorRep);
  }

  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(uy));
  CUDA_CALL(cudaFree(gains));
}
