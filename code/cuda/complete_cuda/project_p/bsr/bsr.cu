template <typename data_type, typename index_type>
__global__ void bcsr_spmv_kernel_thread_per_row_row_major_matrix (
  index_type n_block_rows,
  index_type bs,
  const index_type * __restrict__ col_ids,
  const index_type * __restrict__ row_ptr,
  const data_type * __restrict__ data,
  const data_type * __restrict__ x,
  data_type *y)
{
  const index_type idx = blockIdx.x * blockDim.x + threadIdx.x;
  const index_type row = idx % bs; //row inside the block
  const index_type block_row = idx / bs;//block row
  const index_type first_block = row_ptr[block_row]; // first block of the row
  const index_type last_block = row_ptr[block_row + 1];// last block of the row

  if (row < bs && block_row < n_block_rows)// why not outside the row_ptr
    {

      data_type local_out = 0.0;//keep a sum

      for (index_type block = first_block; block < last_block; block++)// for every block in the block row
        {
	  //get the first coll of the block or else position
          const index_type first_col = col_ids[block] * bs;
          for (index_type col = 0; col < bs; col++)//for all the colomns in the block
            local_out += x[first_col + col] * data[block * bs * bs + row * bs + col];
        }

      y[block_row * bs + row] = local_out;
    }
}
/*
 so what does a thread take

x x x x x x x x x
x x x x x x x x x
0 0 t t 0 0 t t t---<-
0 0 x x 0 0 x x x---
x x x x x x x x x
x x x x x x x x x



 
 */
