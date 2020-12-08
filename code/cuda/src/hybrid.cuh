#include <cuda_runtime.h>

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
    data_type* x, int rows_count,data_type* y,
    int* ell_cols,data_type* ell_data,
    data_type* coo_data,int* coo_row_ids,int* coo_col_ids
{


  /// ELL Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    grid_size.x = (rows_count + block_size.x - 1) / block_size.x;

    ell_spmv_kernel<<<grid_size, block_size>>> (
        rows_count, matrix.ell_matrix->elements_in_rows,ell_cols, matrix.ell_data, x, y);
  }

  /// COO Part
  {
    dim3 block_size = dim3 (512);
    dim3 grid_size {};

    const auto n_elements = matrix.coo_matrix->get_matrix_size ();
    grid_size.x = (n_elements + block_size.x - 1) / block_size.x;

    coo_spmv_kernel<<<grid_size, block_size>>> (
        n_elements, coo_col_ids, matrix.coo_row_ids, matrix.coo_data, x, y);
  }


}
