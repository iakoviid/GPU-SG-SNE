template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_column_by_column (
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type * __restrict__ y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  index_type col = first_block * bs + lane / bs;//why lane/bs
  index_type r = lane % bs;//ok i get the r

  data_type *partial_sums = shared_memory<data_type> ();// lets see 

  data_type local_out = 0.0;

  for (; col < last_block * bs; col += 32 / bs)//what
    {
      const index_type block = col / bs;
      const index_type c = col % bs;

      const data_type value = data[block * bs * bs + c * bs + r];
      const data_type x_value = x[col_ids[block] * bs + c];
      local_out += x_value * value;
    }

  partial_sums[threadIdx.x] = local_out;

  for (index_type stride = round_up_to_power_of_two((32 / bs) / 2); stride > 0; stride /= 2)
    {
      __syncthreads ();
      if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
        partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
    }

  if (lane < bs)
    y[block_row * bs + lane] = partial_sums[threadIdx.x];
}
