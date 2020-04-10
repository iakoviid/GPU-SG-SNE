#include "common.h"
#define H_NUM 3
#define N_GRID_SIZE 137

#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
#define LAGRANGE_INTERPOLATION


#ifdef LAGRANGE_INTERPOLATION

__inline__
double g1(double d){
  return   0.5 * d*d*d - 1.0 * d*d - 0.5   * d + 1;
}

__inline__
double g2(double d){
  double cc = 1.0/6.0;
  return -cc * d*d*d + 1.0 * d*d - 11*cc * d + 1;
}

#else

__inline__
double g1(double d){
  return  1.5 * d*d*d - 2.5 * d*d         + 1;
}

__inline__
double g2(double d){
  return -0.5 * d*d*d + 2.5 * d*d - 4 * d + 2;
}
#endif

double *generateRandomCoord(int n, int d) {

  double *y = (double *)malloc(n * d * sizeof(double));
  srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((double) rand() / (RAND_MAX))*100;

  return y;
}

__global__ ComputeChargesKernel(volatile double* __restrict__ VScat,const double *const y_d,const int n,const int d, const int n_terms){
  for (uint32_t register tid = blockIdx.x * blockDim.x + threadIdx.x; tid < n; tid += blockDim.x * gridDim.x){
    VScat[tid*(d+1)]=1;
    for(int j=0;j<d;j++)
      VScat[tid*(d+1)+j+1]=y_d[tid*d+j];
}

}

void ComputeCharges(double* VScat,double* y_d,n,d){
  int threads=1024;
  int Blocks=64;
  ComputeChargesKernel<<<Blocks,threads>>>(VScat,y_d,n,d,d+1);

}

__global__ void compute_repulsive_forces_kernel(
    volatile double *__restrict__ frep,
    const double *const Y,
    const int num_points, const int nDim, const double *const Phi,
    volatile double *__restrict__ zetaVec)
    {
      register double Ysq=0;
      register double z=0;

    for(register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < num_points;TID+=gridDim.x*blockDim.x){

    for(uint32_t j=0; j<nDim; j++){
        Ysq+=Y[TID*nDim+j]*Y[TID*nDim+j];
        z-=2*Y[TID*nDim+j]*Phi[TID*(nDim+1)+j+1];
    }
    z+=(1+2*Ysq)*Phi[TID*(nDim+1)];
    zetaVec[TID]=z;
    for(uint32_t j=0;j<nDim;j++){
      frep[TID*nDim+j] = Y[TID*nDim+j]*Phi[TID*(nDim+1)]-Phi[TID*(nDim+1)+j+1];
    }}
}
double zetaAndForce(double *Ft_d,double* y_d,int n,int d,double* Phi,thrust::device_vector<double> & zetaVec){

  int threads=1024;
  int Blocks=64;
  compute_repulsive_forces_kernel<<<Blocks,threads>>>(Ft_d,y_d,n,d,Phi,thrust::raw_pointer_cast(zetaVec.data()));
  double z=thrust::reduce(zetaVec.begin(),zetaVec.end(),0.0,thrust::plus<double>);

  return z-n;
}
int getBestGridSize( int nGrid )
{

  // list of FFT sizes that work "fast" with FFTW
  int listGridSize[N_GRID_SIZE] =
    {8,9,10,11,12,13,14,15,16,20,25,26,28,32,33,35,
     36,39,40,42,44,45,48,49,50,52,54,55,56,60,63,64,65,66,70,72,75,
     77,78,80,84,88,90,91,96,98,99,100,104,105,108,110,112,117,120,
     125,126,130,132,135,140,144,147,150,154,156,160,165,168,175,176,
     180,182,189,192,195,196,198,200,208,210,216,220,224,225,231,234,
     240,245,250,252,260,264,270,273,275,280,288,294,297,300,308,312,
     315,320,325,330,336,343,350,351,352,360,364,375,378,385,390,392,
     396,400,416,420,432,440,441,448,450,455,462,468,480,490,495,500,
     504,512};

  // select closest (larger) size for given grid size
  for (int i=0; i<N_GRID_SIZE; i++)
    if ( (nGrid+2) <= listGridSize[i] )
      return listGridSize[i]-2;

  return listGridSize[N_GRID_SIZE-1]-2;

}

void __global__ scalePoints(double* Y,const double maxy,const int nGridDim,const int n, const int d){

  for(register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < num_points*d;TID+=gridDim.x*blockDim.x){
      Y[TID]=Y[TID]*(nGridDim-1)/maxy;
  }

}
void __global__ s21d(double* V,double* y,double* q,double* ng,int nPts,int nDim,int nVec){

  uint32_t f1;
  double d;
  double v1,v2,v3,v4;
  double qv;
  for(register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < num_points*d;TID+=gridDim.x*blockDim.x){
    f1=(uint32_t) floor( y[TID] );
    d=y[TID]-(double)f1;
    v1 = g2(1+d);
    v2 = g1(  d);
    v3 = g1(1-d)  ;
    v4 = g2(2-d);
    for (j=0;j<nVec;j++)
      qv=q[TID+j*nPts];
      V[f1+1+j*nPts]+=qv*v1;
      V[f1+2+j*nPts]+=qv*v2;
      V[f1+3+j*nPts]+=qv*v3;
      V[f1+4+j*nPts]+=qv*v4;

  }


}

void nuConv(double * PhiScat,double *y_d,double* VScat, int n, int d,int nGridDim,double maxy) {

  // ~~~~~~~~~~ scale them from 0 to ng-1
  int threads=1024;
  int blocks=64;
  scalePoints<<<blocks,threads>>>(y_d,maxy,nGridDim,n,d);

  // ~~~~~~~~~~ find exact h
  double h = maxy / (nGridDim - 1 - std::numeric_limits<coord>::epsilon() );

  /*switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD) */
  s2g1d<<<blocks,threads>>>( VGrid, y, VScat, nGridDim+2, n, d, d+1 );
  /*  else
      s2g1drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g2d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g2drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      s2g3d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
      s2g3drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  }*/

}
__global__ generateBoxIdx(uint64_t Code,double* Y,double scale, int nPts,int nDim, int nGrid,double multQuant)
{
  uint64_t C[3];
  uint32_t qLevel = ceil(log(nGrid)/log(2));
  double Yscale;
  for(register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;TID+=gridDim.x*blockDim.x){
    for(int j=0;j<nDim,j++){
      Yscale=Y[TID+j]/scale;
      C[j]=uint32_t abs( floor( multQuant * Yscale ) );
    }
    switch (nDim) {

    case 1:
      Code[TID]= (uint64_t) C[0];

    case 2:
      Code[TID]= ( ( (uint64_t) C[1] ) << qLevel ) |
              ( ( (uint64_t) C[0] )           );

    case 3:
      Code[TID]= ( ( (uint64_t) C[2] ) << 2*qLevel ) |
                ( ( (uint64_t) C[1] ) <<   qLevel ) |
                ( (uint64_t) C[0] )             );


  }

}
void relocateCoarseGrid(double* Y,int nPts, int d,int gridDim,double* y_new,uint32_t* iPerm,uint32_t* ib,uint32_t* cb,uint64_t Codes){
  int threads=1024;
  int blocks=64;
  double multQuant = nGrid - 1 - std::numeric_limits<dataval>::epsilon();
  generateBoxIdx<<<blocks, threads>>>(Codes,Y,maxy,nPts,d,gridDim,multQuant );
  cudaDeviceSynchronize();

  uint32_t qLevel = ceil(log(nGrid)/log(2));


  thrust ::device_ptr<int> kc(keysC);
  thrust ::stable_sort_by_key(kc, kc + lenghtC, make_zip_iterator(make_tuple(Cx_ptr, Cy_ptr, Cz_ptr)));

}
//Watch for template
double ComputeRepulsiveForces(double *Ft, double *y, int n, int d, double h){
    //Tranfer data to the GPU
    double *Ft_d,*y_d;
    CUDA_CALL(cudaMallocManaged(&Ft_d,d*n*sizeof(double)));
    CUDA_CALL(cudaMallocManaged(&y_d,d*n*sizeof(double)));
    CUDA_CALL(cudaMemcpy(y_d,y,n*d*sizeof(double)));
    thrust::device_ptr<double> yVec_ptr(y_d);
    thrust::device_vector<double> yVec_d(yVec_ptr,yVec_ptr+n*d);
    thrust::device_vector<double>::iterator iter=thrust::max_element(yVec_d.begin(),yVec_d.end());
    unsigned int position= iter-yVec_d.begin();
    double maxy=yVec_d[position];

    int nGrid = std::max( (int) std::ceil( maxy / h ), 14 );
    nGrid = getBestGridSize(nGrid);
    std::cout << "Grid: " << nGrid << " h: " << h << "maxy: "<<maxy <<std::endl;

    double *y_to_use;
    CUDA_CALL(cudaMallocManaged(&y_to_use,d*n*sizeof(double)));
    uint32_t *iPerm,*ib,*cb;
    CUDA_CALL(cudaMallocManaged(&iPerm,n*sizeof(uint32_t)));
    CUDA_CALL(cudaMallocManaged(&ib,nGrid*sizeof(uint32_t)));
    CUDA_CALL(cudaMallocManaged(&cb,nGrid*sizeof(uint32_t)));
    uint64_t *Codes;
    CUDA_CALL(cudaMallocManaged(&Codes,n*sizeof(uint64_t)));



    relocateCoarseGrid(y_d,n,d,nGrid,y_to_use,iPerm,ib,cb,Codes);


    //Compute chargesQijt
    double *VScat;
    CUDA_CALL(cudaMallocManaged(&VScat,(d+1)*n*sizeof(double)));
    ComputeCharges(VScat,y_d,n,d);

    double* PhiScat;
    CUDA_CALL(cudaMallocManaged(&PhiScat,n*(d+1)*sizeof(double)));

    nuConv(PhiScat,y_d,VScat,n,d,d+1,nGrid,maxy);


    thrust::device_vector<double> zetaVec(n);
    z=zetaAndForce(Ft_d,y_d,PhiScat,n,d,zetaVec);


    return z;
}



double computeFrepulsive_exact(double *frep, double *pointsX, int N, int d) {

  double *zetaVec = (double *)calloc(N, sizeof(double));

  for(int i = 0; i < N; i++) {
    double Yi[10] = {0};
    for (int dd = 0; dd < d; dd++)
      Yi[dd] = pointsX[i * d + dd];

    double Yj[10] = {0};

    for (int j = 0; j < N; j++) {

      if (i != j) {

        double dist = 0.0;
        for (int dd = 0; dd < d; dd++) {
          Yj[dd] = pointsX[j * d + dd];
          dist += (Yj[dd] - Yi[dd]) * (Yj[dd] - Yi[dd]);
        }

        for (int dd = 0; dd < d; dd++) {
          frep[i * d + dd] += (Yi[dd] - Yj[dd]) / ((1 + dist) * (1 + dist));
        }

        zetaVec[i] += 1.0 / (1.0 + dist);
      }
    }
  }

  double zeta = 0;
  for(int i=0; i<N; i++){
    zeta+=zetaVec[i];
  }

  for (int i = 0; i < N; i++) {
    for(int k=0;k<d;k++){
    frep [(i * d) +k] /= zeta;}
  }

  free(zetaVec);

  return zeta;
}

bool testRepulsiveTerm(int n, int d) {

  bool flag = true;

  double *y = generateRandomCoord(n, d);
  double *Fg = (double *)calloc(n * d, sizeof(double));
  double *Ft = (double *)malloc(n * d * sizeof(double));

  double h[H_NUM] = {0.05, 0.08, 0.13};

  double zg = computeFrepulsive_exact(Fg, y, n, d);

  for (int i = 0; i < H_NUM; i++) {
    for(int k=0;k<n*d;k++){
      Ft[k] = 0.0;
    }
    //double zt=0;
    /*double y_c=(double)malloc(N*d*sizeof(double));
    for(int i=0;i<N;i++){
      for(int j=0;j<d;j++){
        yc[i+N*d]=y[i*d+j];
      }
    }*/
    double zt=computeMine(Ft,y_c,n,d,h[i]);
    //double z = computeFrepulsive_interp(Ft, y, n, d, h[i]);


    double maxErr = 0;
    for (int jj = 0; jj < n * d; jj++)
      maxErr = maxErr < abs(Fg[jj] - Ft[jj]) ? abs(Fg[jj] - Ft[jj]) : maxErr;

    if (maxErr > 1e-2 || abs(zg - zt) / zg > 1e-2)
      flag = false;
  }
  if(flag){
    printf("Succes\n");
  }else{
    //printf("z1=%lf z2=%lf\n",zg,zt );
  }

  free(y);
  free(Fg);
  free(Ft);

  return flag;
}

int main(int argc, char **argv) {
  int N = atoi(argv[1]);
  int d = atoi(argv[2]);
   double *x;
   x=generateRandomCoord(N,d);
  testRepulsiveTerm(N, d);

  free(x);
  return 0;
}
