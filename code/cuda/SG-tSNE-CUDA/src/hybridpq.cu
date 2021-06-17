extern cudaStream_t streamAttr;
extern int Blocks;
extern int Threads;
template <typename data_type, int d>
__global__ void
ell_spmv_kernel(const unsigned int n, const unsigned int elements_in_rows,
                const unsigned int *__restrict__ col_ids, const data_type *__restrict__ data,
                const data_type *__restrict__ Y, data_type *__restrict__ Fatr) {
  register unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;
  register unsigned int column;
  register data_type sum1, sum2, sum3;
  register unsigned int element;
  register data_type dx, dy, dz;
  register data_type value,dist;

//  if (row < n) {
    for (row = blockIdx.x * blockDim.x + threadIdx.x; row < n;
     row += blockDim.x * gridDim.x)  {

    sum1 = 0;
    sum2 = 0;
    sum3 = 0;

    for (element = 0; element < elements_in_rows; element++) {
      const unsigned int element_offset = row + element * n;
      column = col_ids[element_offset];
      switch (d) {
      case 1:
        dx = Y[row] - Y[column];
        dist = dx * dx;
        value = data[element_offset] / (1 + dist);
        sum1 += value * dx;
        break;
      case 2:
        dx = Y[row] - Y[column];
        dy = Y[row + n] - Y[column + n];
        dist = dx * dx + dy * dy;
        value = data[element_offset] / (1 + dist);
        sum1 += value * dx;
        sum2 += value * dy;
        break;
      case 3:
        dx = Y[row] - Y[column];
        dy = Y[row + n] - Y[column + n];
        dz = Y[row + 2 * n] - Y[column + 2 * n];
        dist = dx * dx + dy * dy + dz * dz;
        value = data[element_offset] / (1 + dist);
        sum1 += value * dx;
        sum2 += value * dy;
        sum3 += value * dz;
        break;
      }
    }

    switch (d) {
    case 1:
      Fatr[row] = sum1;
      break;
    case 2:
      Fatr[row] = sum1;
      Fatr[row + n] = sum2;
      break;
    case 3:
      Fatr[row] = sum1;
      Fatr[row + n] = sum2;
      Fatr[row + 2 * n] = sum3;
      break;
    }
  }
}
template <typename data_type, int d>
__global__ void coo_spmv_kernel(const int n_elements,
                                const unsigned int *__restrict__ col_ids,
                                const unsigned int *__restrict__ row_ids,
                                const data_type *__restrict__ data,
                                const data_type *__restrict__ Y,
                                data_type *__restrict__ Fattr, const unsigned int n) {
  register unsigned int row, column;
  register data_type dist, pq;
  register unsigned int element;
  register data_type dx, dy, dz;
  for (element = blockIdx.x * blockDim.x + threadIdx.x; element < n_elements;
       element += blockDim.x * gridDim.x) {
    row = row_ids[element];
    column = col_ids[element];
    switch (d) {
    case 1:
      dx = (Y[row] - Y[column]);
      dist = dx * dx;
      pq = data[element] / (1 + dist);
      atomicAdd(Fattr + row, pq * dx);
      break;
    case 2:
      dx = (Y[row] - Y[column]);
      dy = (Y[row + n] - Y[column + n]);
      dist = dx * dx + dy * dy;
      pq = data[element] / (1 + dist);
      atomicAdd(Fattr + row, pq * dx);
      atomicAdd(Fattr + row + n, pq * dy);
      break;
    case 3:
      dx = (Y[row] - Y[column]);
      dy = (Y[row + n] - Y[column + n]);
      dz = (Y[row + 2 * n] - Y[column + 2 * n]);
      dist = dx * dx + dy * dy + dz * dz;
      pq = data[element] / (1 + dist);
      atomicAdd(Fattr + row, pq * dx);
      atomicAdd(Fattr + row + n, pq * dy);
      atomicAdd(Fattr + row + 2 * n, pq * dz);
      break;
    }
  }
}

template <typename data_type>
void gpu_hybrid_spmv(int elements_in_rows, int coo_size, data_type *Y,
                     unsigned int rows_count, data_type *F,
                     unsigned int *ell_cols, data_type *ell_data,
                     data_type *coo_data, unsigned int *coo_row_ids,
                     unsigned int *coo_col_ids, int d) {

  /// ELL Part
  {
    switch (d) {
    case 1:
      ell_spmv_kernel<data_type, 1><<<Blocks, Threads, 0, streamAttr>>>(
          rows_count, elements_in_rows, ell_cols, ell_data, Y, F);
      break;
    case 2:
      ell_spmv_kernel<data_type, 2><<<Blocks, Threads, 0, streamAttr>>>(
          rows_count, elements_in_rows, ell_cols, ell_data, Y, F);
      break;
    case 3:
      ell_spmv_kernel<data_type, 3><<<Blocks, Threads, 0, streamAttr>>>(
          rows_count, elements_in_rows, ell_cols, ell_data, Y, F);
      break;
    }
  }

  /// COO Part
  {
    const int n_elements = coo_size;

    switch (d) {
    case 1:
      coo_spmv_kernel<data_type, 1><<<Blocks, Threads, 0, streamAttr>>>(
          n_elements, coo_col_ids, coo_row_ids, coo_data, Y, F, rows_count);
      break;
    case 2:
      coo_spmv_kernel<data_type, 2><<<Blocks, Threads, 0, streamAttr>>>(
          n_elements, coo_col_ids, coo_row_ids, coo_data, Y, F, rows_count);
      break;
    case 3:
      coo_spmv_kernel<data_type, 3><<<Blocks, Threads, 0, streamAttr>>>(
          n_elements, coo_col_ids, coo_row_ids, coo_data, Y, F, rows_count);
      break;
    }
  }
}
