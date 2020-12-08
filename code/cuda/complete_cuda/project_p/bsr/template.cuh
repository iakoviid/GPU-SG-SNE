#define index_type int
#define acc double
#define bs 8

__device__ index_type round_up_to_power_of_two (index_type v)
{
  v--;
  v |= v >> 1;
  v |= v >> 2;
  v |= v >> 4;
  v |= v >> 8;
  v |= v >> 16;
  v++;

  return v;
}
/*We could allocate warp per block row in another way. The new algorithm is claimed to be more efficient
 in handling matrices with block sizes that are not powers of two.*/
//  In the algorithm warp iterates through its block row’s columns, covering 32/bs columns at a time.
// The algorithm limits max block size by warp size. Although this optimization would limit matrix block size,
// it might be quite interesting to try.
__global__ void bcsr_spmv_kernel_column_by_column_template (
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const acc * __restrict__ data,
  const acc * __restrict__ x,
  acc * __restrict__ y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type lane = idx % 32;
  const index_type block_row = idx / 32; ///< Warp per block row
  const index_type first_block = row_ptr[block_row];
  const index_type last_block = row_ptr[block_row + 1];

  index_type col = first_block * bs + lane / bs;// this showes where in the value and colomn matrix we are but id bs is 4
  index_type r = lane % bs;// lane/bs = 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 4 4
                          // r 0 1 2 3 0 1 2 3 0 1 2 3

  extern __shared__ acc partial_sums[]; // shared memory array for every thread

  acc local_out = 0.0;

  for (; col < last_block * bs; col += 32 / bs)
    {
      const index_type block = col / bs;
      const index_type c = col % bs;

      const acc value = data[block * bs * bs + c * bs + r];
      const acc x_value = x[col_ids[block] * bs + c];
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
