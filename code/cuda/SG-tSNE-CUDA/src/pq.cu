#include "pq.cuh"
#include <thrust/reduce.h>
#include <float.h>
extern cudaStream_t streamAttr;
extern int Blocks;
extern int Threads;
template <class dataPoint>
void AttractiveEstimation(int *row, int *col, dataPoint *data, dataPoint *Fattr,
                          dataPoint *y, int n, int d, int blockSize,
                          int blockRows, int num_blocks, int nnz, int format) {

  if (format == 0) {
    pq_csr_naive<<<Blocks, Threads>>>(row, col, data, n, d, y, Fattr);

  } else if (format == 1) {
    // bsr_col_pq(data, row, col, y, Fattr, blockRows, blockSize, num_blocks, n,
    // d);
  } else if (format == 2) {
    coo_pq_kernel<<<Blocks, Threads>>>(nnz, col, row, data, y, Fattr, d, n);
  } else if (format == 3) {
    printf("hybrid\n");
  }
}
template <class dataPoint>
void AttractiveEstimation(sparse_matrix<dataPoint> P, int d, dataPoint *y,
                          dataPoint *Fattr) {

  if (P.format == 0) {
    pq_csr_naive<dataPoint, int>
        <<<Blocks, Threads, 0, streamAttr>>>(P.row, P.col, P.val, P.n, d, y, Fattr);

  } else if (P.format == 1) {
    // bsr_col_pq<dataPoint>(P.val, P.row, P.col, y, Fattr, P.blockRows,
    // P.blockSize, P.nnzb,
    //         P.n, d);
  } else if (P.format == 2) {
    switch (d) {
    case 1:
      coo_pq_kernel<dataPoint, 1><<<Blocks, Threads, 0, streamAttr>>>(
          P.nnz, P.col, P.row, P.val, y, Fattr, P.n);
      break;
    case 2:
      coo_pq_kernel<dataPoint, 2><<<Blocks, Threads, 0, streamAttr>>>(
          P.nnz, P.col, P.row, P.val, y, Fattr, P.n);
      break;
    case 3:
      coo_pq_kernel<dataPoint, 3><<<Blocks, Threads, 0, streamAttr>>>(
          P.nnz, P.col, P.row, P.val, y, Fattr, P.n);
      break;
    }

  } else if (P.format == 3) {
    gpu_hybrid_spmv<dataPoint>(P.elements_in_rows, P.coo_size, y, P.n, Fattr,
                               P.ell_cols, P.ell_data, P.coo_data,
                               P.coo_row_ids, P.coo_col_ids, d);
  }
}



template <typename data_type,int d>
__global__ void
ComputeKLkernel(const int n_elements, const matidx *__restrict__ col_ids,
              const matidx *__restrict__ row_ids,
              const data_type *__restrict__ data,
              const data_type *__restrict__ Y, data_type *__restrict__ Cij, const int n,data_type zeta,data_type alpha) {
  register matidx row, column;
  register data_type dist, p,q;
  register unsigned int element;
  register data_type dx, dy, dz;
  for (element = blockIdx.x * blockDim.x + threadIdx.x; element < n_elements;
       element += blockDim.x * gridDim.x) {
    row = row_ids[element];
    column = col_ids[element];
    switch (d) {
    case 1:
      dx = (Y[row] - Y[column]);
      dist = dx * dx;
      p = data[element]*alpha;
      q= 1/(zeta* (1 + dist));
      Cij[element]=p*log((p+FLT_MIN)/(q+FLT_MIN));
      break;
    case 2:
      dx = (Y[row] - Y[column]);
      dy = (Y[row + n] - Y[column + n]);
      dist = dx * dx + dy * dy;
      p = data[element]*alpha;
      q= 1/(zeta* (1 + dist));
      Cij[element]=p*log((p+FLT_MIN)/(q+FLT_MIN));
      break;
    case 3:
      dx = (Y[row] - Y[column]);
      dy = (Y[row + n] - Y[column + n]);
      dz = (Y[row + 2 * n] - Y[column + 2 * n]);
      dist = dx * dx + dy * dy + dz * dz;
      p = data[element]*alpha;
      q= 1/(zeta* (1 + dist));
      Cij[element]=p*log((p+FLT_MIN)/(q+FLT_MIN));
      break;
    }
  }
}

template<class dataPoint>
dataPoint tsneCost(sparse_matrix<dataPoint>P,dataPoint* y, int n,int d,dataPoint alpha,dataPoint zeta ){
dataPoint* Cij;
  gpuErrchk(cudaMallocManaged(&Cij, P.nnz* sizeof(dataPoint)));
switch(d){
case 1:
	ComputeKLkernel<dataPoint,1><<<Blocks, Threads>>>(P.nnz, P.col, P.row, P.val, y, Cij, P.n,zeta,alpha);
	break;
case 2:
        ComputeKLkernel<dataPoint,2><<<Blocks, Threads>>>(P.nnz, P.col, P.row, P.val, y, Cij, P.n,zeta,alpha);
        break;
case 3:
        ComputeKLkernel<dataPoint,3><<<Blocks, Threads>>>(P.nnz, P.col, P.row, P.val, y, Cij, P.n,zeta,alpha);
        break;


}
cudaDeviceSynchronize();
dataPoint C= thrust::reduce(Cij, Cij+P.nnz);
gpuErrchk(cudaFree(Cij));
return C;
}
template
float tsneCost(sparse_matrix<float> P,float* y, int n,int d,float alpha,float zeta );

template
double tsneCost(sparse_matrix<double> P,double* y, int n,int d,double alpha,double zeta );
template void AttractiveEstimation(sparse_matrix<float> P, int d, float *y,
                                   float *Fattr);
template void AttractiveEstimation(sparse_matrix<double> P, int d, double *y,
                                   double *Fattr);
