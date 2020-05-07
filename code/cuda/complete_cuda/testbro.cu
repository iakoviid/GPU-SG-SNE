#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
typedef double coord;
__device__ __host__ static inline double sign(double x) {

  return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
}

template <class dataPoint>
void compute_dyCPU(dataPoint *const dy, dataPoint const *const Fattr,
                dataPoint const *const Frep, int const N, int const dim,
                dataPoint const alpha) {

  for (int i = 0; i < N; i++) {
    for (int d = 0; d < dim; d++) {
      dy[i * dim + d] = (alpha * Fattr[i * dim + d] - Frep[i * dim + d]);
    }
  }
}
__global__ void compute_dy(coord* dy,coord* Fattr,coord* Frep,int n,int d,coord alpha){
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n*d;
       TID += gridDim.x * blockDim.x) {
         dy[TID]=(alpha*Fattr[TID])-Frep[TID];

       }
}

template <class dataPoint>
void update_positionsCPU(dataPoint *const dY, dataPoint *const uY, int const N,
                         int const no_dims, dataPoint *const Y,
                         dataPoint *const gains, double const momentum,
                         double const eta) {

  // Update gains
  for (int i = 0; i < N * no_dims; i++) {
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    if (gains[i] < .01)
      gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];
  }
  // find mean
  dataPoint meany[no_dims];
  for (int i = 0; i < no_dims; i++) {
    meany[i] = 0;
  }
  for (int i = 0; i < no_dims; i++) {
    for (int j = 0; j < N; j++) {
      meany[i] += Y[j * no_dims + i];
    }


    meany[i] /= N;
  }

  // zero-mean
  for(int i=0;i<no_dims;i++){
    printf("host Mean %lf\n",meany[i] );
  }

  for (int n = 0; n < N; n++) {
    for (int d = 0; d < no_dims; d++) {
      Y[n * no_dims + d] -= meany[d];
    }
  }
}
__global__ void gradient_update(coord* dY,coord* uY,int N,int no_dims,coord* Y,coord* gains,coord momentum,coord eta){
  for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < N * no_dims; i+=gridDim.x*blockDim.x) {
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);

    if (gains[i] < .01)
      gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];

  }}
  template <class dataPoint>
__global__ void addScalar(dataPoint* a,dataPoint scalar,uint32_t length){
  for (int i = threadIdx.x+blockIdx.x*blockDim.x; i < length; i+=gridDim.x*blockDim.x) {
    a[i]+=scalar;
}

}
 void update_positions(coord* dY,coord* uY,int n,int d,coord* Y,coord* gains,coord momentum,coord eta){

   gradient_update<<<1,1>>>( dY,uY, n, d, Y, gains,momentum, eta);
   coord* meany=(coord*)malloc(d*sizeof(coord));

   thrust::device_ptr<double> yVec_ptr = thrust::device_pointer_cast(Y);

   for(int i=0;i<d;i++){
     meany[i]=thrust::reduce(yVec_ptr+(i)*n,yVec_ptr+(i+1)*n)/n;
     printf("cuda mean %lf \n",meany[i] );
     addScalar<<<32,256>>>(&Y[i*n],-meany[i],n);
   }



}

double *generateRandomCoord(int n, int d) {

  double *y = (double *)malloc(n * d * sizeof(double));
  srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((double)rand() / (RAND_MAX)) * 100;

  return y;
}

template <class dataPoint>
dataPoint* copydata(dataPoint *const w,int n,int d){
  dataPoint* v=(dataPoint *)malloc(sizeof(dataPoint)*n*d);
  for(int i=0;i<n;i++){
    for(int j=0;j<d;j++){

      v[i+n*j]=w[i*d+j];

    }
  }
return v;
}

template <class dataPoint>
void compair(dataPoint *const w,dataPoint* v,dataPoint*dv,int n,int d){
  int bro=1;
  cudaMemcpy(v,dv,d*n*sizeof(dataPoint), cudaMemcpyDeviceToHost);

  printf("--------------------------------------------------------------------------------\n" );
  printf("----------------------------------Compair---------------------------------------\n" );
  for(int i=0;i<n;i++){
    for(int j=0;j<d;j++){
      if(abs(w[i*d+j]-v[i+j*n])<0.00001){
      //printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
    }else{
        bro=0;
        printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);}

    }
  }
  if(bro==1){printf("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Success~~~~~~~~~~~~~~~~~~~~~~~~\n" );}
  printf("--------------------------------------------------------------------------------\n" );

}

int main(int argc, char **argv) {
  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  coord* dy=(coord *)malloc(n*d*sizeof(coord));
  coord* Fattr=generateRandomCoord(n,d);
  coord* Frep=generateRandomCoord(n,d);
  coord a=12;
  compute_dyCPU(dy,Fattr,Frep,n,d,a);

  coord* Fd1=(coord *)copydata(Fattr,n,d);
  coord* Fd2=(coord *)copydata(Frep,n,d);

  coord* Fad;
  coord* Frd;
  cudaMallocManaged(&Fad,n*d*sizeof(coord));
  cudaMallocManaged(&Frd,n*d*sizeof(coord));
  cudaMemcpy(Fad,Fd1,d*n*sizeof(coord), cudaMemcpyHostToDevice);
  cudaMemcpy(Frd,Fd2,d*n*sizeof(coord), cudaMemcpyHostToDevice);

  coord* ddy;
  cudaMallocManaged(&ddy,n*d*sizeof(coord));
  compute_dy<<<32,256>>>(ddy,Fad,Frd,n,d,a);
  coord* dcy=(coord *)malloc(n*d*sizeof(coord));

  compair(dy,dcy,ddy,n,d);
  coord* uY=generateRandomCoord(n,d);
  coord* gains=generateRandomCoord(n,d);
  coord* Y=generateRandomCoord(n,d);
  coord momentum=0.5;
  coord eta=0.5;

    coord* uYd=(coord *)copydata(uY,n,d);
    coord* gainsd=(coord *)copydata(gains,n,d);
    coord* Y_d=(coord *)copydata(Y,n,d);
    coord* duY,*dgains,*dY_d;
    cudaMallocManaged(&duY,n*d*sizeof(coord));
    cudaMallocManaged(&dgains,n*d*sizeof(coord));
    cudaMallocManaged(&dY_d,n*d*sizeof(coord));

      cudaMemcpy(dY_d,Y_d,d*n*sizeof(coord), cudaMemcpyHostToDevice);
      cudaMemcpy(dgains,gainsd,d*n*sizeof(coord), cudaMemcpyHostToDevice);
      cudaMemcpy(duY,uYd,d*n*sizeof(coord), cudaMemcpyHostToDevice);
    compair(uY,uYd,duY,n,d);
    compair(gains,gainsd,dgains,n,d);
    compair(Y,Y_d,dY_d,n,d);

  update_positionsCPU( dy,uY,n,d,Y, gains, momentum,eta);


  update_positions(ddy,duY,n,d,dY_d, dgains, momentum,eta);
  printf("--------------------------Gains-----------------------\n" );
  compair(gains,gainsd,dgains,n,d);
  printf("--------------------------Gainz-----------------------\n" );

  compair(Y,Y_d,dY_d,n,d);

}
