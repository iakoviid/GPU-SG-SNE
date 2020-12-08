#include "matrix_converter.h"
#include <stdio.h>
#include <iostream>
#include <memory>
#include <cuda_runtime.h>
#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

template <typename data_type>
__global__ void ell_spmv_kernel (
    unsigned int n_rows,
    unsigned int elements_in_rows,
    const unsigned int *col_ids,
    const data_type*data,
    const data_type*x,
    data_type*y)
{
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n_rows)
  {
    data_type dot = 0;
    for (unsigned int element = 0; element < elements_in_rows; element++)
    {
      const unsigned int element_offset = row + element * n_rows;
      dot += data[element_offset] * x[col_ids[element_offset]];
    }
    y[row] = dot;
  }
}
template <typename data_type>
__global__ void coo_spmv_kernel (
    unsigned int n_elements,
    const unsigned int *col_ids,
    const unsigned int *row_ids,
    const data_type*data,
    const data_type*x,
    data_type*y)
{
  unsigned int element = blockIdx.x * blockDim.x + threadIdx.x;

  if (element < n_elements)
  {
    const data_type dot = data[element] * x[col_ids[element]];
    atomicAdd (y + row_ids[element], dot);
  }
}

template <typename data_type>
void gpu_hybrid_spmv (
    const hybrid_matrix_class<data_type> &matrix,
    data_type* x, unsigned int rows_count,data_type* y,
    unsigned int* ell_cols,data_type* ell_data,
    data_type* coo_data,unsigned int* coo_row_ids,unsigned int* coo_col_ids)
{


  /// ELL Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (rows_count + block_size.x - 1) / block_size.x;

    ell_spmv_kernel<<<grid_size, block_size>>> (
        rows_count, matrix.ell_matrix->elements_in_rows,ell_cols, ell_data, x, y);
  }

  /// COO Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    const auto n_elements = matrix.coo_matrix->get_matrix_size ();
    grid_size.x = (n_elements + block_size.x - 1) / block_size.x;

    coo_spmv_kernel<<<grid_size, block_size>>> (
        n_elements, coo_col_ids,coo_row_ids, coo_data, x, y);
  }


}
using namespace std;
int main(){
	csr_matrix_class<float> A;
	A.data.reset (new float[590]);
	A.columns.reset (new unsigned int[590]);
	A.row_ptr.reset (new unsigned int[101]);
	for(int i=0;i<100;i++){
for(int j=0;j<5;j++){
	A.data[i*5+j]=1;
	A.columns[i*5+j]=i+j;

}A.row_ptr[i]=5*i;

}
	for(int i=0;i<90;i++){
		A.data[500+i]=1;
		A.columns[500+i]=i;
	}

	A.nnz=590;
	A.n=100;
	int n=100;
	int nnz=500;
	A.row_ptr[100]=590;
	coo_matrix_class<float> B(A);
	ell_matrix_class<float> C(A);

	hybrid_matrix_class<float> D(A);
	D.allocate(A,0.001);
	float* cooy;
	float* hybridy;
	float* x_h,*x;
	x_h=(float* )malloc(n*sizeof(float));
	for(int i=0;i<n;i++){
		x_h[i]=i;
	}
	CUDA_CALL(cudaMalloc(&x, n * sizeof(float)));
	CUDA_CALL(cudaMalloc(&cooy, n * sizeof(float)));
	CUDA_CALL(cudaMalloc(&hybridy, n * sizeof(float)));
	cudaMemcpy(x, x_h, n * sizeof (float), cudaMemcpyHostToDevice);

	unsigned int* col,*row;
	float* val;
	CUDA_CALL(cudaMalloc(&col, nnz * sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc(&row, nnz * sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc(&val, nnz * sizeof(float)));
	dim3 block_size = dim3 (512);
	dim3 grid_size {};
	const auto n_elements =nnz;
	grid_size.x = (n_elements + block_size.x - 1) / block_size.x;
	cudaMemcpy (val, B.data.get (), nnz * sizeof (float), cudaMemcpyHostToDevice);
	cudaMemcpy (col, B.cols.get (), nnz * sizeof (unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy (row, B.rows.get (), nnz * sizeof (unsigned int), cudaMemcpyHostToDevice);

	coo_spmv_kernel<<<grid_size, block_size>>> (nnz, col, row,val, x, cooy);

	cudaFree(col);
	cudaFree(row);
	cudaFree(val);
	float* result=(float *)malloc(sizeof(float)*n);
	cudaMemcpy(result, cooy, n * sizeof (float), cudaMemcpyDeviceToHost);
	cudaFree(cooy);

	unsigned int* ell_cols,* coo_col_ids,*coo_row_ids;
	float* ell_data, *coo_data;
	const size_t A_size = D.ell_matrix->get_matrix_size ();
	const size_t col_ids_size = A_size;
	CUDA_CALL(cudaMalloc(&ell_data, A_size * sizeof(float)));
	CUDA_CALL(cudaMalloc(&ell_cols, A_size * sizeof(unsigned int)));
	cudaMemcpy(ell_data, D.ell_matrix->data.get (), A_size * sizeof (float), cudaMemcpyHostToDevice);
	cudaMemcpy(ell_cols, D.ell_matrix->columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

	const size_t coo_size = D.coo_matrix->get_matrix_size ();
	CUDA_CALL(cudaMalloc(&coo_data, coo_size * sizeof(float)));
	CUDA_CALL(cudaMalloc(&coo_col_ids, coo_size * sizeof(unsigned int)));
	CUDA_CALL(cudaMalloc(&coo_row_ids, coo_size * sizeof(unsigned int)));
	cudaMemcpy (coo_data, D.coo_matrix->data.get (), coo_size * sizeof (float), cudaMemcpyHostToDevice);
	cudaMemcpy (coo_col_ids, D.coo_matrix->cols.get (), coo_size * sizeof (unsigned int), cudaMemcpyHostToDevice);
	cudaMemcpy (coo_row_ids, D.coo_matrix->rows.get (), coo_size * sizeof (unsigned int), cudaMemcpyHostToDevice);


	gpu_hybrid_spmv(D,x,n,hybridy,ell_cols,ell_data,coo_data,coo_row_ids,coo_col_ids);
	float* result2=(float *)malloc(sizeof(float)*n);
	cudaMemcpy(result2, hybridy, n * sizeof (float), cudaMemcpyDeviceToHost);
	cudaFree(hybridy);

	for(int i=0;i<n;i++){
		float er=(result[i]-result2[i])*(result[i]-result2[i]);
		if(er>0.001){

			printf("i=%d %f vs %f\n",i,result[i],result2[i] );
		}
	}
	cudaFree(x);
	free(result);
	free(result2);
	free(x_h);

return 0;
}
