#include "pq.cuh"
void bsr_col_pq(coord *val, int *block_ptr, int *col_ind, coord *y, coord *F,
                int n, int blockSize, int num_blocks, int rows, int d) {

  dim3 block_size = 32;
  dim3 grid_size{};

  grid_size.x = (n * 32 + block_size.x - 1) / block_size.x;

  switch (blockSize) {
  case 1:
    bsr_col<coord, int, 1>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  case 2:
    bsr_col<coord, int, 2>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  case 3:
    bsr_col<coord, int, 3>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  case 4:
    bsr_col<coord, int, 4>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  case 8:
    bsr_col<coord, int, 8>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  case 16:
    bsr_col<coord, int, 16>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  case 32:
    bsr_col<coord, int, 32>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind, block_ptr, val, y, F, rows, d);
    break;
  }
}

void AttractiveEstimation(int *row, int *col, coord *data,
                          coord  *Fattr, coord *y, int n,
                          int d, int blockSize,int blockRows,int num_blocks,int nnz,int format) {

  if (format == 0) {
    pq_csr_naive<<<32, 256>>>(row, col, data,n, d, y, Fattr);

  } else if (format == 1) {
    bsr_col_pq(data, row, col, y, Fattr, blockRows, blockSize, num_blocks, n, d);
  } else if (format == 2) {
    coo_pq_kernel<<<32,512>>>( nnz, col, row, data, y, Fattr, d, n);
  } else if (format == 3) {
    printf("hybrid\n");
  }
}
