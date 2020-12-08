template <typename data_type, typename index_type, index_type bs>
__global__ void bcsr_spmv_kernel_warp_per_row_row_major_matrix (
  index_type n_block_rows,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type *y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = (idx / 32) % bs;//warp in a row
  const index_type lane = idx % 32;
  const index_type block_row = (idx / 32) / bs;//warp in a block row
  const index_type first_block = row_ptr[block_row];//first and last block in the block row
  const index_type last_block = row_ptr[block_row + 1];

  data_type local_out = 0.0;

  if (row < bs && block_row < n_block_rows)
    {
      for (index_type loc_col = lane; loc_col < bs * (last_block - first_block); loc_col += 32)//the warp does the sum
        {//dont understand block niether 
          const index_type block = first_block + loc_col / bs;
          const index_type c = loc_col % bs;
          const index_type col = col_ids[block] * bs + c;
          local_out += x[col] * data[block * bs * bs + row * bs + c];
        }
    }

  local_out = warp_reduce (local_out);

  if (row < bs && block_row < n_block_rows && lane == 0)
    y[block_row * bs + row] = local_out;
}
