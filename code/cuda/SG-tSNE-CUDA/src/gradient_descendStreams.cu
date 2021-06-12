#include "compute_error.cu"
#include "gradient_descend.cuh"
#include <fstream>
#include <math.h>
#include <thrust/device_ptr.h>
#include <thrust/execution_policy.h>
cudaStream_t streamAttr = 0;
cudaStream_t streamRep = 0;
using namespace std;
template <class dataPoint>
__global__ void compute_dy(volatile dataPoint *__restrict__ dy,const dataPoint * const Fattr,const dataPoint * const Frep,
                          const int n,const int d,const dataPoint alpha) {
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
__global__ void gradient_update(const dataPoint *const dY,volatile dataPoint * __restrict__ uY,const int N,
                                const int no_dims,volatile dataPoint *__restrict__ Y, volatile dataPoint *__restrict__ gains,
                                const dataPoint momentum,const dataPoint eta) {
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
}

template <class dataPoint, class Complext>
dataPoint compute_gradient(dataPoint *Fattr, dataPoint *Frep,
    dataPoint *dy, double *timeFrep, double *timeFattr, tsneparams params,
    dataPoint *y, sparse_matrix<dataPoint> P, double *TotaltimeInfo,
    cufftHandle &plan, cufftHandle &plan_rhs, int nGrid, dataPoint *yt,
    dataPoint *VScat, dataPoint *PhiScat, dataPoint *VGrid, dataPoint *PhiGrid,
    Complext *Kc, Complext *Xc, thrust::device_vector<dataPoint> &zetaVec,
     dataPoint *errorRep = nullptr) {
   double timeInfo[7] = {0};
  // ----- parse input parameters
  int d = params.d;
  int n = params.n;
  // ----- timing
  struct GpuTimer timer;

  // ------ Compute PQ (fattr)
  timer.Start();
  AttractiveEstimation(P, d, y, Fattr);
  timer.Stop();
  *timeFattr = timer.Elapsed();

  cudaDeviceSynchronize();

  // ------ Compute QQ (frep)
  timer.Start();
//  dataPoint* yh=(dataPoint*)malloc(sizeof(dataPoint)*n*d);
//CUDA_CALL(cudaMemcpy(yh, y, n * d * sizeof(dataPoint),cudaMemcpyDeviceToHost));
 dataPoint zeta = computeFrepulsive_interp(
      Frep, y, n, d, (dataPoint)params.h, timeInfo, nGrid, plan, plan_rhs, yt,
      VScat, PhiScat, VGrid, PhiGrid, Kc, Xc, zetaVec);
//  dataPoint* Freph=(dataPoint *)malloc(sizeof(dataPoint)*n*d);

//CUDA_CALL(cudaMemcpy(Freph, Frep, n * d * sizeof(dataPoint),cudaMemcpyDeviceToHost));
 // free(yh);
  //free(Freph);

  timer.Stop();
  *timeFrep += timer.Elapsed();
  timeInfo[0]=*timeFattr;
  for (int i = 0; i < 7; i++) {TotaltimeInfo[i] = timeInfo[i];}

  // ----- Compute gradient (dY)
  cudaDeviceSynchronize();
  compute_dy<<<64, 1024>>>(dy, Fattr, Frep, n, d, (dataPoint)params.alpha);
cudaDeviceSynchronize();
  if (params.ComputeError > 0 && errorRep != nullptr) {
    *errorRep = computeError(Frep, y, n, d);
  }



  return zeta;
}
void kl_minimization(float *y, tsneparams params, sparse_matrix<float> P,double* timeInfo) {

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  float momentum = .5, final_momentum = .8;
  float eta = 200.0;
  // int iterPrint = 50;
  // cudaStreamCreate(&streamAttr);
  // cudaStreamCreate(&streamRep);
  double timeFattr = 0.0;
  double timeFrep = 0.0;
  double timeUpdate=0.0;
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

  //double timeInfo[1000 * 7] = {0};

  cufftHandle plan, plan_rhs;

  int n1 = getBestGridSize(params.ng);
  int n2 = n1 + 2;
  switch (d) {
  case 1: {
    int ng[1] = {(int)n2};
    cufftPlan1d(&plan, n2, CUFFT_C2C, 1);
    cufftPlanMany(&plan_rhs, 1, ng, NULL, 1, n2, NULL, 1, n2, CUFFT_C2C, d + 1);
    break;
  }
  case 2: {
        int ng[2] = {(int)n2, (int)n2};
    int *inembed = NULL, *onembed = NULL;
      cufftPlan2d(&plan, n2, n2, CUFFT_C2C);
//cufftPlanMany(&plan, 2, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 1);
    cufftPlanMany(&plan_rhs, 2, ng, inembed, 1, n2 * n2, onembed, 1, n2 * n2,
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
  // cufftSetStream(plan, streamRep);
  // cufftSetStream(plan_rhs, streamRep);
  int m = d + 1;
  int nVec = d + 1;
  /*Allocate memory*/
  float *yt;
  CUDA_CALL(cudaMallocManaged(&yt, (d)*n * sizeof(float)));
  float *VScat;
  CUDA_CALL(cudaMallocManaged(&VScat, (d + 1) * n * sizeof(float)));
  float *PhiScat;
  CUDA_CALL(cudaMallocManaged(&PhiScat, (d + 1) * n * sizeof(float)));
  int szV = pow(n1 + 2, d) * m;
  float *VGrid;
  CUDA_CALL(cudaMallocManaged(&VGrid, szV * sizeof(float)));
  float *PhiGrid;
  CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(float)));
  Complex *Kc, *Xc;
  CUDA_CALL(cudaMallocManaged(&Kc, szV * sizeof(Complex)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * szV * sizeof(Complex)));
  thrust::device_vector<float> zetaVec(n);
  float *Fattr;
  float *Frep;
  CUDA_CALL(cudaMallocManaged(&Fattr, n * d * sizeof(float)));
  CUDA_CALL(cudaMallocManaged(&Frep, n * d * sizeof(float)));
  struct timeval t1, t2;
  gettimeofday(&t1, NULL);
  struct GpuTimer timer;
  float KL[100]={0};
  for (int iter = 0; iter < max_iter; iter++) {

    initKernel<<<64, 1024>>>(Fattr, (float)0.0, n * d);
    initKernel<<<64, 1024, 0, streamRep>>>(VGrid, (float)0, szV);
    initKernel<<<64, 1024, 0, streamRep>>>(PhiGrid, (float)0, szV);
cudaDeviceSynchronize();
    if (iter % 100 == 0) {
      printf("---------------------------%d---------------------------\n",
             iter);
//      appendProgressGPU(y, n,d,"sg_dump.txt");
	std::cout<<"Zeta= "<<zeta<<"\n";

    }

    if (params.ComputeError > 0 && iter % 10 == 0 ) {
      zeta = compute_gradient(Fattr,Frep,dy, &timeFrep, &timeFattr, params, y, P,
                              &timeInfo[7*iter], plan, plan_rhs, n1, yt,
                              VScat, PhiScat, VGrid, PhiGrid, Kc, Xc, zetaVec,
                              &errorRep[(int)iter / 10]);
    } else {
      zeta = compute_gradient(Fattr,Frep,dy, &timeFrep, &timeFattr, params, y, P,
                              &timeInfo[7*iter], plan, plan_rhs, n1, yt,
                              VScat, PhiScat, VGrid, PhiGrid, Kc, Xc, zetaVec);
    }
/*
   if(iter%10==0){ KL[iter/10]=tsneCost(P,y,  n,d,params.alpha, zeta );
 std::cout<<"Error : "<<KL[iter/10]<<"\n";
}*/
    timer.Start();
    update_positions(dy, uy, n, d, y, gains, momentum, eta);
    timer.Stop();
    timeUpdate += timer.Elapsed();

    // Stop lying about the P-values after a while, and switch momentum
    if (iter == stop_lying_iter) {
      params.alpha = 1;
    }

    // Change momentum after a while
    if (iter == mom_switch_iter) {
      momentum = final_momentum;
    }
  }
thrust::device_ptr<float> yVec_ptr(y);
float meany[4];
  for (int i = 0; i < d; i++) {
    meany[i] = thrust::reduce(yVec_ptr + (i)*n, yVec_ptr + (i + 1) * n) / n;
cudaDeviceSynchronize();
  addScalar<<<32, 256>>>(y+i*n, -meany[i], n);
  }
cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("time=%lf\n",elapsedTime );
  timeInfo[7*1000]=elapsedTime;
/*
  cout << "Time for computing Attractive Estimation " << timeFattr << "ms\n";
  cout << "Time for computing Repulsive Estimation " << timeFrep << " ms\n";
  cout << "Time for computing point update " << timeUpdate << " ms\n";
  cout << "Detailed " << timeInfo[1] << " ms in s2g, " << timeInfo[2]
       << " ms in g2g, " << timeInfo[3] << " ms in g2s\n";
  cout << timeInfo[4] << " ms in zetaAndForce, " << timeInfo[5]
       << " ms  in nuconv, " << timeInfo[6] << " ms in preprocessing\n";
  cout << "and " << timeInfo[0] << " in permutations\n";
*/
//  timeInfo[0]=timeFattr;
/*
  ofstream fout_cost;
  fout_cost.open ("sg_cost.txt");
  for(int i=0;i<100;i++){fout_cost<<KL[i]<<"\n"; }
  fout_cost.close();
*/
  if (params.ComputeError > 0) {
    ofstream errorf;
    ifstream errorfin;
    errorfin.open("errorInfo.txt");
    errorf.open("errorInfo.txt",std::ios::app);
     if(errorfin.is_open())
   {
    for (int i = 0; i < errorCalcs; i++) {
      errorf << errorRep[i] << " ";
    }
	}
    errorf<<"\n";

    errorf.close();
    errorfin.close();
    free(errorRep);
  }
  CUDA_CALL(cudaFree(PhiGrid));
  CUDA_CALL(cudaFree(VGrid));
  CUDA_CALL(cudaFree(yt));
  CUDA_CALL(cudaFree(VScat));
  CUDA_CALL(cudaFree(PhiScat));
  CUDA_CALL(cudaFree(Kc));
  CUDA_CALL(cudaFree(Xc));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(uy));
  CUDA_CALL(cudaFree(gains));
  CUDA_CALL(cudaFree(Fattr));
  CUDA_CALL(cudaFree(Frep));

}

void kl_minimization(double *y, tsneparams params, sparse_matrix<double> P) {
  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;
  // int iterPrint = 50;
  cudaStreamCreate(&streamAttr);
  cudaStreamCreate(&streamRep);
  double timeFattr = 0.0;
  double timeFrep = 0.0;
  double timeUpdate=0.0;

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
  // double timeInfo[7] = {0};
  double timeInfo[1000 * 7] = {0};

  cufftHandle plan, plan_rhs;

  int n1 = getBestGridSize(params.ng);
  int n2 = n1 + 2;
  switch (d) {
  case 1: {
    int ng[1] = {(int)n2};
    cufftPlan1d(&plan, n2, CUFFT_Z2Z, 1);
    cufftPlanMany(&plan_rhs, 1, ng, NULL, 1, n2, NULL, 1, n2, CUFFT_Z2Z, d + 1);
    break;
  }
  case 2: {
    int ng[2] = {(int)n2, (int)n2};
    cufftPlanMany(&plan, 2, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
    cufftPlanMany(&plan_rhs, 2, ng, NULL, 1, n2 * n2, NULL, 1, n2 * n2,
                  CUFFT_Z2Z, d + 1);
    break;
  }
  case 3: {
    int ng[3] = {(int)n2, (int)n2, (int)n2};
    cufftPlanMany(&plan, 3, ng, NULL, 1, 0, NULL, 1, 0, CUFFT_Z2Z, 1);
    cufftPlanMany(&plan_rhs, 3, ng, NULL, 1, n2 * n2 * n2, NULL, 1,
                  n2 * n2 * n2, CUFFT_Z2Z, d + 1);
    break;
  }
  }
  cufftSetStream(plan, streamRep);
  cufftSetStream(plan_rhs, streamRep);
  int m = d + 1;
  int nVec = d + 1;
  /*Allocate memory*/
  double *yt;
  CUDA_CALL(cudaMallocManaged(&yt, (d)*n * sizeof(double)));
  double *VScat;
  CUDA_CALL(cudaMallocManaged(&VScat, (d + 1) * n * sizeof(double)));
  double *PhiScat;
  CUDA_CALL(cudaMallocManaged(&PhiScat, (d + 1) * n * sizeof(double)));
  int szV = pow(n1 + 2, d) * m;
  double *VGrid;
  CUDA_CALL(cudaMallocManaged(&VGrid, szV * sizeof(double)));
  double *PhiGrid;
  CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(double)));
  ComplexD *Kc, *Xc;
  CUDA_CALL(cudaMallocManaged(&Kc, szV * sizeof(ComplexD)));
  CUDA_CALL(cudaMallocManaged(&Xc, nVec * szV * sizeof(ComplexD)));
  thrust::device_vector<double> zetaVec(n);
  double *Fattr;
  double *Frep;
  CUDA_CALL(cudaMallocManaged(&Fattr, n * d * sizeof(double)));
  CUDA_CALL(cudaMallocManaged(&Frep, n * d * sizeof(double)));
  struct GpuTimer timer;

  for (int iter = 0; iter < max_iter; iter++) {
    initKernel<<<64, 1024>>>(Fattr, (double)0.0, n * d);
    initKernel<<<64, 1024, 0, streamRep>>>(VGrid, (double)0, szV);
    initKernel<<<64, 1024, 0, streamRep>>>(PhiGrid, (double)0, szV);
    if (iter % 100 == 0) {
      printf("---------------------------%d---------------------------\n",
             iter);
    }

    if (params.ComputeError > 0 && iter % 10 == 0) {
      zeta = compute_gradient(Fattr,Frep,dy, &timeFrep, &timeFattr, params, y, P,
                              &timeInfo[iter * 7], plan, plan_rhs, n1, yt,
                              VScat, PhiScat, VGrid, PhiGrid, Kc, Xc, zetaVec,
                              &errorRep[(int)iter / 10]);
    } else {
      zeta = compute_gradient(Fattr,Frep,dy, &timeFrep, &timeFattr, params, y, P,
                              &timeInfo[iter * 7], plan, plan_rhs, n1, yt,
                              VScat, PhiScat, VGrid, PhiGrid, Kc, Xc, zetaVec);
    }

    timer.Start();
    update_positions(dy, uy, n, d, y, gains, momentum, eta);
    timer.Stop();
    timeUpdate+= timer.Elapsed();
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
  cout << "Time for computing point update " << timeUpdate << " ms\n";
  ofstream fout;  // Create Object of Ofstream
  ifstream fin;
  fin.open("Attractive.txt");
  fout.open ("Attractive.txt",ios::app); // Append mode
  if(fin.is_open())
    {fout<<P.n<<" "<<timeFattr<<" "<<P.nnz <<"\n";}
    fin.close();
    fout.close(); // Closing the file
  cout << "Detailed " << timeInfo[1] << " ms in s2g, " << timeInfo[2]
       << " ms in g2g, " << timeInfo[3] << " ms in g2s\n";
  cout << timeInfo[4] << " ms in zetaAndForce, " << timeInfo[5]
       << " ms  in nuconv, " << timeInfo[6] << " ms in preprocessing\n";
  cout << "and " << timeInfo[0] << " in permutations\n";

  ofstream myfile;
  myfile.open("timeInfo.txt");
  for (int i = 0; i < max_iter; i++) {
    for (int j = 0; j < 7; j++) {
      myfile << timeInfo[7 * i + j] << " ";
    }
    myfile << "\n";
  }
  myfile.close();

  if (params.ComputeError > 0) {
    ofstream errorf;
    errorf.open("errorInfo.txt");
    for (int i = 0; i < errorCalcs; i++) {
      errorf << errorRep[i] << "\n";
    }
    errorf.close();
    free(errorRep);
  }
  CUDA_CALL(cudaFree(PhiGrid));
  CUDA_CALL(cudaFree(VGrid));
  CUDA_CALL(cudaFree(yt));
  CUDA_CALL(cudaFree(VScat));
  CUDA_CALL(cudaFree(PhiScat));
  CUDA_CALL(cudaFree(Kc));
  CUDA_CALL(cudaFree(Xc));
  CUDA_CALL(cudaFree(dy));
  CUDA_CALL(cudaFree(uy));
  CUDA_CALL(cudaFree(gains));
  CUDA_CALL(cudaFree(Fattr));
  CUDA_CALL(cudaFree(Frep));
}
