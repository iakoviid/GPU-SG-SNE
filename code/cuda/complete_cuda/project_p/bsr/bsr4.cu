template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_thread_per_row_column_major_matrix_coal_x (
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type *y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = idx % bs;
  const index_type block_row = idx / bs;
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  data_type *cache_x = shared_memory<data_type> ();

  cache_x[threadIdx.x] = 0.0;
  data_type local_out = 0.0;

  for (index_type block = first_block; block < last_block; block++)
    {
      __syncwarp ();
      if (threadIdx.x < bs)
        cache_x[threadIdx.x] = x[col_ids[block] * bs + threadIdx.x];
      __syncwarp ();

      for (index_type col = 0; col < bs; col++)
        local_out += cache_x[col] * data[block * bs * bs + col * bs + row];
    }

  if (row < bs)
    y[block_row * bs + row] = local_out;
}

/*
 so what does a thread take
                       

x x x x x x x x x      y
x x x x x x x x x      y
0 0 t t 0 0 t t t---<- y
0 0 x x 0 0 x x x---   y
x x x x x x x x x      y
x x x x x x x x x      y




 */

