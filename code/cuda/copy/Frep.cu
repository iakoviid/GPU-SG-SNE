
#include "Frep.cuh"

template <class dataPoint>
__global__ void ArrayCopy(dataPoint* a,dataPoint* b,uint32_t n)
{
  uint32_t tidx = threadIdx.x + blockDim.x * blockIdx.x;
  uint32_t stride = blockDim.x * gridDim.x;
  for (; tidx < n; tidx += stride){
    b[tidx]=a[tidx];
 }
}

template <class dataPoint>
  __global__ void ArrayScale(dataPoint *a, dataPoint scalar, uint32_t length) {
  for (int i = threadIdx.x + blockIdx.x * blockDim.x; i < length;
       i += gridDim.x * blockDim.x) {
    a[i] *= scalar;
  }
}
__global__ void ComputeChargesKernel(coord *__restrict__ VScat,
                                     const coord *const y_d, const int n,
                                     const int d, const int n_terms) {

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    for (int j = 0; j < d; j++) {
      VScat[TID + (j + 1) * n] = y_d[TID + (j)*n];
      // if(threadIdx.x==0){printf("y_d[%d]=%lf\n",TID+(j)*n ,y_d[TID+(j)*n]);}
    }
    VScat[TID] = 1;
  }
}
void ComputeCharges(coord *VScat, coord *y_d, int n, int d) {
  int threads = 1024;
  int Blocks = 64;
  ComputeChargesKernel<<<Blocks, threads>>>(VScat, y_d, n, d, d + 1);
}

__global__ void compute_repulsive_forces_kernel(
    volatile coord *__restrict__ frep, const coord *const Y,
    const int num_points, const int nDim, const coord *const Phi,
    volatile coord *__restrict__ zetaVec, uint32_t *iPerm) {
   register coord Ysq = 0;
   register coord z = 0;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;TID < num_points; TID += gridDim.x * blockDim.x) {
    Ysq=0;
    z=0;
    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[TID + j * num_points] * Y[TID + j * num_points];
      z -= 2 * Y[TID + j * num_points] * Phi[TID + (num_points) *(j + 1)];
      frep[iPerm[TID] + j * num_points]=  Y[TID + j * num_points]*Phi[TID]-Phi[TID+(j+1)*num_points];
    }

    z += (1 + 2 * Ysq) * Phi[TID];
    zetaVec[TID] = z;
  }
}


coord zetaAndForce(coord *Ft_d, coord *y_d, int n, int d, coord *Phi,
                   uint32_t *iPerm,
                   thrust::device_vector<coord> &zetaVec) {
// can posibly reduce amongs the threads and then divide

  int threads = 1024;
  int Blocks = 64;
  compute_repulsive_forces_kernel<<<Blocks, threads>>>(Ft_d, y_d, n, d, Phi, thrust::raw_pointer_cast(zetaVec.data()),iPerm);
  coord z = thrust::reduce(zetaVec.begin(), zetaVec.end()) - n;
  ArrayScale<<<Blocks, threads>>>(Ft_d,1/z,n*d);
  return z;
}

coord computeFrepulsive_interp(coord *Frep, coord *y, int n, int d, coord h) {


  // ~~~~~~~~~~ make temporary data copies
  coord* yr,*yt;
  CUDA_CALL(cudaMallocManaged(&yr, (d) * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&yt, (d) * n * sizeof(coord)));

  // ~~~~~~~~~~ move data to (0,0,...)
  thrust::device_ptr<coord> yVec_ptr(y);
  thrust::device_vector<coord> yVec_d(yVec_ptr, yVec_ptr + n * d);
  unsigned int position;
  coord *miny = (coord *)malloc(sizeof(coord) * d);
  for (int j = 0; j < d; j++) {

    thrust::device_vector<coord>::iterator iter = thrust::min_element(
        yVec_d.begin() + j * n, yVec_d.begin() + n * (j + 1));
    position = iter - (yVec_d.begin());
    miny[j] = yVec_d[position];
    addScalar<<<32, 256>>>(&y[j * n], -miny[j], n);
  }

  // ~~~~~~~~~~ find maximum value (across all dimensions) and get grid size
  //--G I have something similar max(maxy/h,14) vs max((maxy-miny)*2,20)

  thrust::device_vector<coord>::iterator iter =
      thrust::max_element(yVec_d.begin(), yVec_d.end());
  position = iter - yVec_d.begin();
  coord maxy = yVec_d[position];
  int nGrid = std::max((int)std::ceil(maxy / h), 14);
  nGrid = getBestGridSize(nGrid);

  ArrayCopy<<<32,256>>>(y,yt,n*d);
  //printf("maxy=%lf\n",maxy );

  uint32_t *ib;
  uint32_t *cb;
  CUDA_CALL(cudaMallocManaged(&ib, nGrid* sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, nGrid* sizeof(uint32_t)));
  thrust::device_vector<uint32_t> iPerm(n);
  thrust::sequence(iPerm.begin(), iPerm.end());


  relocateCoarseGrid(yt,iPerm,ib,cb,n,nGrid,d);



  coord *VScat;
  coord *PhiScat;
  CUDA_CALL(cudaMallocManaged(&VScat, (d+1) * n * sizeof(coord)));
  CUDA_CALL(cudaMallocManaged(&PhiScat, (d+1) * n * sizeof(coord)));
  ComputeCharges(VScat, yt, n, d);
  ArrayCopy<<<32,256>>>(yt,yr,n*d);


  nuconv(PhiScat,yt, VScat, ib, cb, n, d, d+1, nGrid);
/*
  coord* PhiScat_h =(coord *) malloc(sizeof(coord)*n*(d+1));
  cudaMemcpy(PhiScat_h, PhiScat, (d+1) * n * sizeof(coord), cudaMemcpyDeviceToHost);
  printf("PhiScatGPU=%lf %lf %lf %lf %lf \n",PhiScat_h[0],PhiScat_h[1],PhiScat_h[2],PhiScat_h[3] );
  */
  thrust::device_vector<coord> zetaVec(n);
  coord zeta = zetaAndForce(Frep, yr, n, d, PhiScat, thrust::raw_pointer_cast(iPerm.data()),zetaVec);




  CUDA_CALL(cudaFree(yr));
  CUDA_CALL(cudaFree(yt));
  CUDA_CALL(cudaFree(VScat));
  CUDA_CALL(cudaFree(PhiScat));
  CUDA_CALL(cudaFree(ib));
  CUDA_CALL(cudaFree(cb));
  return zeta;
}