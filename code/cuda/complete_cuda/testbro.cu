#include "gridding.cuh"
#include "gridding.hpp"
#include "relocateData.cuh"
#include "relocateData.hpp"
#include "non_periodic_conv.cuh"
#include "non_periodic_conv.hpp"
#include "nuconv.hpp"
#include "nuconv.cuh"
#include "Frep.cuh"
#include "Frep.hpp"
#include "gradient_descend.hpp"
#include "gradient_descend.cuh"
#include <iostream>
#include "tsne.cuh"

using namespace std;
typedef double coord;
#include "matrix_indexing.hpp"

#define idx2(i,j,d) (SUB2IND2D(i,j,d))

coord *generateRandomCoord(int n, int d) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));
  srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((coord)rand() / (RAND_MAX)) * 100;

  return y;
}

template <class dataPoint>
void copydata(dataPoint *const w, dataPoint *dw, int n, int d) {
  dataPoint *v = (dataPoint *)malloc(sizeof(dataPoint) * n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {

      v[i + n * j] = w[i * d + j];
    }
  }
  cudaMemcpy(dw, v, d * n * sizeof(dataPoint), cudaMemcpyHostToDevice);
  free(v);
  return;
}

template <class dataPoint>
void compair(dataPoint *const w, dataPoint *dv, int n, int d,const  char *message,
             int same) {
  int bro = 1;
  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  printf(
      "----------------------------------Compair %s----------------------------"
      "-----------\n",
      message);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if (same == 0) {
        if (abs(w[i * d + j] - v[i + j * n]) < 0.01) {
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else {
          bro = 0;
          cout << "Error "
               << "Host=" << w[i * d + j] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      } else {
        if (abs(w[i + j * n] - v[i + j * n]) < 0.01 ){
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else {
          bro = 0;
          cout << "Error "
               << "Host=" << w[i + j * n] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      }
    }
  }
  if (bro == 1) {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Success~~~~~~~~~~~~~~~~~~~~~~~~\n");
  } else {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Error~~~~~~~~~~~~~~~~~~~~~~~~\n");
  }
  free(v);
}

template <class dataPoint>
  __global__ void Add2(dataPoint *a, dataPoint* b,dataPoint*c, uint32_t d,uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
         for(int j=0;j<d;j++){
           c[i+j*length]=a[i+j*length]+b[i+j*length];
         }
  }
}

template <class dataPoint>
  __global__ void Add1(dataPoint *a, dataPoint* b,dataPoint*c, uint32_t d,uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
         for(int j=0;j<d;j++){
           c[i*d+j]=a[i*d+j]+b[i*d+j];
         }
  }
}

int main(int argc, char **argv) {
  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  coord *y, *y_d;

  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  uint32_t *ib, *cb, *ib_h, *cb_h;
  CUDA_CALL(cudaMallocManaged(&ib, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, ng * sizeof(uint32_t)));
  ib_h=(uint32_t *)malloc(ng*sizeof(uint32_t));
  cb_h=(uint32_t *)malloc(ng*sizeof(uint32_t));
  thrust::device_vector<uint32_t> iPerm(n);
  thrust::sequence(iPerm.begin(), iPerm.end());
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  for (int i = 0; i < n; i++) {
    iPerm_h[i] = i;
  }
  relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);

  compair(y, y_d, n, d, "Y", 0);

  printf("2 dimensions~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" );
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
    for(int i=0; i<n*d;i++){
      y[i] /= maxy;

    if (1 == y[i])
      y[i] = y[i] - std::numeric_limits<coord>::epsilon();

    y[i] *= (ng-1);
    }
    int szV = pow(ng + 2, d) * (d + 1);

    coord *VGrid = (coord *)calloc(szV, sizeof(coord));
    coord *VGrid_d;
    CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));

    coord *VScat = generateRandomCoord(n, d + 1);
    coord *VScat_d;
    CUDA_CALL(cudaMallocManaged(&VScat_d, (d + 1) * n * sizeof(coord)));
    copydata(VScat, VScat_d, n, d + 1);

    s2g2dCpu(VGrid, y, VScat, ng+2, 1,n,d,d+1);
    s2g2d<<<32, 256>>>(VGrid_d, y_d, VScat_d, ng + 2, n, d, d + 1, maxy);

    coord* VGridh= (coord *)calloc(szV, sizeof(coord));
    cudaMemcpy(VGridh, VGrid_d, szV * sizeof(coord), cudaMemcpyDeviceToHost);
    /*
    for(int j=0;j<d+1;j++){

    for(int k=0;k<ng+2;k++)
    {
    for(int i=0;i<ng + 2;i++){

        //printf("i=%d k=%d j=%d VGrid= %lf  vs VGrid_d= %lf\n",i,k,j,VGrid[SUB2IND4D((i), (k), (j), (0), ng+2, ng+2, d+1)],VGridh[k*(ng+2)+i+j*(ng+2)*(ng+2)] );

      }
    }}
    for(int i=0;i<szV;i++){
      //printf("%lf   vs    %lf\n",VGrid[i],VGridh[i] );
    }
    */
    compair(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "VGrid", 1);
    int nVec=d+1;
    coord* Phi=(coord *)malloc(n*nVec*sizeof(coord));
    coord* Phi_d;
    CUDA_CALL(cudaMallocManaged(&Phi_d,nVec*n*sizeof(coord)));
    g2s1dCpu(Phi,VGrid,y,ng+2,n,d,nVec);
    g2s1d<<<32,256>>>(Phi_d,VGrid_d,y_d,ng+2,n,d,nVec);

    compair(Phi, Phi_d, n, d + 1, "Phi", 0);

    coord *PhiGrid = static_cast<coord *>(calloc(szV, sizeof(coord)));
    coord *PhiGrid_d;
    CUDA_CALL(cudaMallocManaged(&PhiGrid_d, szV * sizeof(coord)));
    uint32_t *const nGridDims = new uint32_t[d]();
    for (int i = 0; i < d; i++) {
      nGridDims[i] = ng + 2;
    }
    coord h = maxy / (ng - 1 - std::numeric_limits<double>::epsilon());

    printf("~~~~~~~~~~~~~~~~~FFT-tests~~~~~~~~~~~~~~~~\n" );


    conv2dnopad(PhiGrid, VGrid, h, nGridDims, d+1, d, 1);
    conv2dnopadcuda(PhiGrid_d, VGrid_d, h, nGridDims, d+1, d);
    uint32_t tpoints = pow(ng + 2, d);


    compair(PhiGrid, PhiGrid_d, tpoints, d + 1, "PhiFFT", 1);

  return 0;
/*
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;

  coord h = maxy / (ng - 1 - std::numeric_limits<double>::epsilon());
  tsneparams params;
  params.h=h;
  params.d=1;
  params.n=n;
  params.alpha=12;
  params.maxIter=10;
  params.earlyIter=5;
  params.np=1;
  coord *dy_d, *dy;
  double timeFrep,timeFattr;
  dy=(coord *)malloc(n*d*sizeof(coord));
  CUDA_CALL(cudaMallocManaged(&dy_d, (d)*n * sizeof(coord)));

  compair(y, y_d, n, d, "KL", 0);

  params.h=h;
  params.d=1;
  params.n=n;
  params.alpha=12;
  params.maxIter=10;
  params.earlyIter=5;
  params.np=1;

  printf("KL kl_minimization~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~\n" );
  kl_minimization( y_d,  params);
  kl_minimizationCPU( y, params);
  compair(y, y_d, n, d, "KL", 0);


    compute_gradient(dy_d, &timeFrep, &timeFattr,params, y_d);
    compute_gradientCPU( dy,&timeFrep,&timeFattr,params, y);
    compair(dy, dy_d, n, d, "dY", 0);

  //realloc--------------------------------------------------------------------
    thrust::device_vector<uint32_t> iPerm(n);
    thrust::sequence(iPerm.begin(), iPerm.end());
    uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
    for (int i = 0; i < n; i++) {
      iPerm_h[i] = i;
    }
    uint32_t *ib, *cb, *ib_h, *cb_h;
    CUDA_CALL(cudaMallocManaged(&ib, ng * sizeof(uint32_t)));
    CUDA_CALL(cudaMallocManaged(&cb, ng * sizeof(uint32_t)));
    ib_h=(uint32_t *)malloc(ng*sizeof(uint32_t));
    cb_h=(uint32_t *)malloc(ng*sizeof(uint32_t));

    relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
    relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
    compair(y, y_d, n, d, "Y", 0);
    compair(iPerm_h, thrust::raw_pointer_cast(iPerm.data()), n,1, "iPerm", 0);



  int szV = pow(ng + 2, d) * (d + 1);
  coord *VGrid = (coord *)calloc(szV, sizeof(coord));
  coord *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));



  //compair(y, y_d, n, d, "Y", 0);
  /*
  compair(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "VGrid", 1);

  int nVec=d+1;
  coord* Phi=(coord *)malloc(n*nVec*sizeof(coord));
  coord* Phi_d;
  CUDA_CALL(cudaMallocManaged(&Phi_d,nVec*n*sizeof(coord)));
  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = ng + 2;
  }
  nuconvCPU(Phi,y, VScat,ib_h, cb_h,n,d,nVec,1,nGridDims[0]);
  nuconv(Phi_d,y_d, VScat_d, ib, cb, n, d, nVec, ng+2);
  compair(Phi, Phi_d, n, d + 1, "nuconv ", 0);
  */
  //coord *Frep, *Frep_d;
  //Frep=(coord *)malloc(n*d*sizeof(coord));
  //CUDA_CALL(cudaMallocManaged(&Frep_d, (d)*n * sizeof(coord)));
  //coord zeta1=computeFrepulsive_interpCPU(Frep, y,  n,  d, h,1);
  //coord zeta2= computeFrepulsive_interp(Frep_d, y_d, n, d, h);
  //printf("zeta1=%lf vs zeta2=%lf\n",zeta1,zeta2 );
  //compair(Frep, Frep_d, n, d, "Frep", 0);
/*
  compair(y, y_d, n, d, "Y", 0);
  coord *VScat = (coord *)malloc(n * (d + 1) * sizeof(coord));

  for (int i = 0; i < n; i++) {

    VScat[i * (d + 1)] = 1.0;
    for (int j = 0; j < d; j++)
      VScat[i * (d + 1) + j + 1] = y[i * d + j];
  }
  coord *VScat_d;
  CUDA_CALL(cudaMallocManaged(&VScat_d, (d+1) * n * sizeof(coord)));
  ComputeCharges(VScat_d, y_d, n, d);
  compair(VScat, VScat_d, n, d+1, "Charges", 0);
  compair(y, y_d, n, d, "Y", 0);

  coord *Frep, *Frep_d;
  Frep=(coord *)calloc(n*d,sizeof(coord));
  CUDA_CALL(cudaMallocManaged(&Frep_d, (d)*n * sizeof(coord)));
  coord zeta1=computeFrepulsive_interpCPU(Frep, y,  n,  d, h,1);
  coord zeta2= computeFrepulsive_interp(Frep_d, y_d, n, d, h);
  printf("zeta1=%lf vs zeta2=%lf\n",zeta1,zeta2 );
  compair(Frep, Frep_d, n, d, "Frep", 0);
  compair(y, y_d, n, d, "Y", 0);
  printf("multidimensional arrays cuda\n" );
  coord* a, *b,*c,*a_d,*b_d;
  n=1<<15;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  CUDA_CALL(cudaMallocManaged(&c, (20)*n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&a_d, (20)*n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&b_d, (20)*n * sizeof(coord)));
  for(int d=1;d<20;d++){
  printf("d=%d  ",d );
  a = generateRandomCoord(n, d);
  b = generateRandomCoord(n, d);
  copydata(a, a_d, n, d);
  copydata(b, b_d, n, d);


  cudaEventRecord(start);
  Add1<<<32,1024>>>(a_d,b_d,c,n,d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Add1 milliseconds %f", milliseconds);

  cudaEventRecord(start);
  Add2<<<32,1024>>>(a_d,b_d,c,n,d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
   milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("vs Add2 milliseconds %f\n", milliseconds);


  free(a);
  free(b);
  }
  printf("matrix_indexing\n" );
int papa=10;
  for(int i=0;i<5;i++){
    for(int j=0;j<papa;j++){
        printf("Idiot %d World %d macros %d transpose %d\n",i+j*5,i*papa+j, SUB2IND2D((i), (j), 5),SUB2IND2D((j), (i), papa) );
        printf("%d\n",idx2(i,j,5) );


  }

}


/*
  coord *y, *y_d;
  int nVec=d+1;
  uint32_t tpoints = pow(ng + 2, d);

  uint32_t *const nGridDims = new uint32_t[d]();

  for (int i = 0; i < d; i++) {
    nGridDims[i] = ng + 2;
  }
  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  thrust::device_vector<uint32_t> iPerm(n);
  thrust::sequence(iPerm.begin(), iPerm.end());
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  for (int i = 0; i < n; i++) {
    iPerm_h[i] = i;
  }
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  uint32_t *ib, *cb, *ib_h, *cb_h;
  CUDA_CALL(cudaMallocManaged(&ib, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&ib_h, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb_h, ng * sizeof(uint32_t)));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Relocation 1 milliseconds %f\n", milliseconds);

  cudaEventRecord(start);

  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Relocation 1 milliseconds %f\n", milliseconds);



  int szV = pow(ng + 2, d) * (d + 1);

  coord *VGrid = (coord *)calloc(szV, sizeof(coord));
  coord *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));

  coord *VScat = generateRandomCoord(n, d + 1);
  coord *VScat_d;
  CUDA_CALL(cudaMallocManaged(&VScat_d, (d + 1) * n * sizeof(coord)));
  copydata(VScat, VScat_d, n, d + 1);
  coord* Phi=(coord *)malloc(n*nVec*sizeof(coord));
  coord* Phi_d;
  CUDA_CALL(cudaMallocManaged(&Phi_d,nVec*n*sizeof(coord)));
  printf("~~~~~~~~~~~~~~~~~~~~~NU conv -------  Test\n" );
  compair(y, y_d, n, d, "Y", 0);
  nuconvCPU(Phi,y, VScat,ib_h, cb_h,n,d,nVec,1,nGridDims[0]);
  nuconv(Phi_d,y_d, VScat_d, ib, cb, n, d, nVec, ng);
  //compair(Phi, Phi_d, n, d + 1, "nuconv ", 0);

  cudaEventRecord(start);
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for (int i = 0; i < n * d; i++) {
    y[i] /= maxy;

    // ~~~~~~~~~~ scale them from 0 to ng-1

    if (1 == y[i]) {
      y[i] = y[i] - std::numeric_limits<coord>::epsilon();
    }

    y[i] *= (ng - 1);
  }
  coord h = maxy / (ng - 1 - std::numeric_limits<double>::epsilon());

  s2g1dCpu(VGrid, y, VScat, ng + 2, 1, n, d, d + 1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Transform 1 milliseconds %f\n", milliseconds);

  cudaEventRecord(start);

  s2g1d<<<32, 512>>>(VGrid_d, y_d, VScat_d, ng + 2, n, d, d + 1, maxy);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Transform 1 milliseconds %f\n", milliseconds);

  //compair(y, y_d, n, d, "Y", 0);
  //compair(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "VGrid", 1);
  //compair(iPerm_h, thrust::raw_pointer_cast(iPerm.data()), n,1, "iPerm", 0);


  g2s1dCpu(Phi,VGrid,y,ng,n,d,nVec);
  g2s1d<<<32,256>>>(Phi_d,VGrid_d,y_d,ng,n,d,nVec);

  //compair(Phi, Phi_d, n, d + 1, "Phi", 0);
  uint32_t m=d+1;


  coord *PhiGrid = static_cast<coord *>(calloc(szV, sizeof(coord)));
  coord *PhiGrid_d;
  CUDA_CALL(cudaMallocManaged(&PhiGrid_d, szV * sizeof(coord)));

  printf("~~~~~~~~~~~~~~~~~FFT-tests~~~~~~~~~~~~~~~~\n" );


  conv1dnopad(PhiGrid, VGrid, h, nGridDims, m, d, 1);
  conv1dnopadcuda(PhiGrid_d, VGrid_d, h, ng + 2, m, d);


  //compair(PhiGrid, PhiGrid_d, tpoints, d + 1, "PhiFFT", 1);
*/


}
