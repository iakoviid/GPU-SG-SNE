#include "types.hpp"
__global__ void hybridpq(matidx n, matidx nz, matidx elements_in_rows, const matidx *ell_col_ids,const matidx *col_ids,
                      const matidx *row_ids,const matval *elldata,const matval* data, const matval *Y, matval *Fattr,
                      int d) {
  unsigned int row = blockIdx.x * blockDim.x + threadIdx.x;

  if (row < n) {
    matval sum1 = 0;
    matval sum2 = 0;
    matval sum3 = 0;
    for (unsigned int element = 0; element < elements_in_rows; element++) {
      const unsigned int element_offset = row + element * n;
      matidx column = ell_col_ids[element_offset];
      coord dist = 0;
      for (int dim = 0; dim < d; dim++) {
        dist += (Y[row + dim * n] - Y[column + dim * n]) *
                (Y[row + dim * n] - Y[column + dim * n]);
      }
      const matval value = elldata[element_offset] / (1 + dist);
      switch (d) {
      case 1:
        sum1 += value * (Y[row] - Y[column]);
        break;
      case 2:
        sum1 += value * (Y[row] - Y[column]);
        sum2 += value * (Y[row + n] - Y[column + n]);
        break;
      case 3:
        sum1 += value * (Y[row] - Y[column]);
        sum2 += value * (Y[row + n] - Y[column + n]);
        sum3 += value * (Y[row + 2 * n] - Y[column + 2 * n]);
        break;
      }
    }
    switch (d) {
    case 1:
      atomicAdd(Fattr + row, sum1);
      break;
    case 2:
      atomicAdd(Fattr + row, sum1);
      atomicAdd(Fattr + row+n, sum2);

        Fatr[row + n] = sum2;
      break;
    case 3:
      atomicAdd(Fattr + row, sum1);
      atomicAdd(Fattr + row+n, sum2);
      atomicAdd(Fattr + row+n*2, sum3);
      break;
    }
  }
  for(unsigned int element=row;element<nz;element+=blockDim.x*gridDim.x){
    data_type dist = 0;
    uint32_t coorow = row_ids[element];
    uint32_t column = col_ids[element];
    for (int dim = 0; dim < d; dim++) {
      dist += (Y[coorow + dim * n] - Y[column + dim * n]) *
              (Y[coorow + dim * n] - Y[column + dim * n]);
    }
    data_type pq = data[element] / (1 + dist);
    switch (d) {
    case 1:
      atomicAdd(Fattr + coorow, pq * (Y[coorow] - Y[column]));
      break;
    case 2:
      atomicAdd(Fattr + coorow, pq * (Y[coorow] - Y[column]));
      atomicAdd(Fattr + coorow + n, pq * (Y[coorow + n] - Y[column + n]));

      break;
    case 3:
      atomicAdd(Fattr + coorow, pq * (Y[coorow] - Y[column]));
      atomicAdd(Fattr + coorow + n, pq * (Y[coorow + n] - Y[column + n]));
      atomicAdd(Fattr + coorow + 2 * n, pq * (Y[coorow + 2 * n] - Y[column + 2 * n]));

      break;
    }
  }



}
template <typename data_type>
void gpu_hybrid_pq(const hybrid_matrix_class<data_type> &matrix, data_type *Y,
                     unsigned int rows_count, data_type *F,
                     unsigned int *ell_cols, data_type *ell_data,
                     data_type *coo_data, unsigned int *coo_row_ids,
                     unsigned int *coo_col_ids, int d) {

  /// ELL Part
  {
    dim3 block_size = dim3(512);
    dim3 grid_size{};

    grid_size.x = (rows_count + block_size.x - 1) / block_size.x;

    ell_spmv_kernel<<<grid_size, block_size>>>(
        rows_count, matrix.ell_matrix->elements_in_rows, ell_cols, ell_data, Y,
        F, d);
  }


}

matrix_rows_statistic get_rows_statistics(unsigned int rows_count,
                                          const unsigned int *row_ptr) {
  matrix_rows_statistic statistic{};
  statistic.min_elements_in_rows = std::numeric_limits<unsigned int>::max() - 1;

  unsigned int sum_elements_in_rows = 0;
  for (unsigned int row = 0; row < rows_count; row++) {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];

    if (elements_in_row > statistic.max_elements_in_rows)
      statistic.max_elements_in_rows = elements_in_row;

    if (elements_in_row < statistic.min_elements_in_rows)
      statistic.min_elements_in_rows = elements_in_row;

    sum_elements_in_rows += elements_in_row;
  }

  statistic.avg_elements_in_rows = sum_elements_in_rows / rows_count;
  statistic.elements_in_rows_std_deviation = 0.0;

  for (unsigned int row = 0; row < rows_count; row++) {
    const auto elements_in_row = row_ptr[row + 1] - row_ptr[row];
    statistic.elements_in_rows_std_deviation += std::pow(
        static_cast<double>(elements_in_row) - statistic.avg_elements_in_rows,
        2);
  }
  statistic.elements_in_rows_std_deviation =
      std::sqrt(statistic.elements_in_rows_std_deviation / rows_count);

  return statistic;
}
