
#define bs 8

#define FULL_WARP_MASK 0xFFFFFFFF

template <class T>
__device__ T warp_reduce (T val)
{
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync (FULL_WARP_MASK, val, offset);

  return val;
}

template <typename coord, typename matidx>
__global__ void bsr1 (
  matidx n_block_rows,
  const matidx * __restrict__ col_ids,
  const matidx * __restrict__ row_ptr,
  const coord * __restrict__ data,
  const coord * __restrict__ x,
  coord *y)
{
  const matidx idx = blockIdx.x * blockDim.x + threadIdx.x;
  const matidx row = (idx / 32) % bs;
  const matidx lane = idx % 32;
  const matidx block_row = (idx / 32) / bs;
  const matidx first_block = row_ptr[block_row];
  const matidx last_block = row_ptr[block_row + 1];

  coord local_out = 0.0;

  if (row < bs && block_row < n_block_rows)
    {
      for (matidx loc_col = lane; loc_col < bs * (last_block - first_block); loc_col += 32)
        {
          const matidx block = first_block + loc_col / bs;
          const matidx c = loc_col % bs;
          const matidx col = col_ids[block] * bs + c;
          local_out += x[col] * data[block * bs * bs + row * bs + c];
        }
    }

  local_out = warp_reduce (local_out);

  if (row < bs && block_row < n_block_rows && lane == 0)
    y[block_row * bs + row] = local_out;
}
int main(int argc, char **argv) {
  const int dim = 1 << atoi(argv[1]);
  int blockSize=1<< atoi(argv[2]);
  COOArrays coo;
  coo.m = dim;
  coo.nnz = dim;
  double errors[4] = {0, 0, 0, 0};
  double alltimes[4] = {0, 0, 0, 0};
  //double flops[3] = {0, 0, 0};
  //flops[0] = coo.nnz * 2;
  int nz=coo.nnz;
  coo.val = (double *)malloc(sizeof(double)*nz);
  coo.rowind = (int *)malloc(sizeof(int)*nz);
  coo.colind = (int *)malloc(sizeof(int)*nz);
  /*eye matrix*/

  for (int idxRow = 0; idxRow < dim; ++idxRow) {
    coo.val[idxRow] = 1.0;
    coo.rowind[idxRow] = idxRow;
    coo.colind[idxRow] = idxRow;}

  CRSArrays crs;
  COO_to_CRS(coo, &crs);
  BCRSArrays bcsr;
  CRS_to_BCRS(crs, &bcsr, blockSize);
  compute_BSRkernel(bcsr,x,y)


  return 0;
}
