#include "matrix_converter.h"
#include <stdio.h>
#include <iostream>
#include <memory>
#include "hybrid.cuh"
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
		A.columns[500+i]=i+105;
	}

	A.nnz=590;
	A.n=100;
	int n=100;
	int nnz=500;
	A.row_ptr[100]=590;
	coo_matrix_class<float> B(A);
	for(int i=0;i<10;i++){
	cout<<B.rows[i]<<" "<<B.cols[i]<<" "<<B.data[i]<<"\n";
	}
	ell_matrix_class<float> C(A);
	for(int i=0;i<10;i++){
	for(int j=0;j<C.elements_in_rows;j++){
		cout<<C.columns[i]<<" "<<C.data[i]<<" ";
}
cout<<"\n";
}
	hybrid_matrix_class<float> D(A);
	D.allocate(A,0.001);
	float* cooy;
	float* hybridy;
	float* x_h,*x;
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
	const auto n_elements = B->get_matrix_size ();
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
	const size_t A_size = matrix.ell_matrix->get_matrix_size ();
	const size_t col_ids_size = A_size;
	CUDA_CALL(cudaMalloc(&ell_data, nnz * sizeof(A_size)));
	CUDA_CALL(cudaMalloc(&ell_cols, A_size * sizeof(unsigned int)));
	cudaMemcpy(ell_data, D.ell_matrix->data.get (), A_size * sizeof (float), cudaMemcpyHostToDevice);
	cudaMemcpy(ell_cols, D.ell_matrix->columns.get (), col_ids_size * sizeof (unsigned int), cudaMemcpyHostToDevice);

	const size_t coo_size = matrix.coo_matrix->get_matrix_size ();
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
		printf("%f vs %f\n",result[i],result2[i] );
	}
	cudaFree(x);
	free(result);
	free(result2);
	free(x_h);

return 0;
}
