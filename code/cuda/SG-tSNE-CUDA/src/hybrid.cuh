#ifndef HYBRID_CUH
#define HYBRID_CUH
#include "common.cuh"
#include "types.hpp"
template <typename data_type, int d>
__global__ void
ell_spmv_kernel(const unsigned int n, const unsigned int elements_in_rows,
                const unsigned int *__restrict__ col_ids, const data_type *__restrict__ data,
                const data_type *__restrict__ Y, data_type *__restrict__ Fatr);

template <typename data_type, int d>
__global__ void coo_spmv_kernel(const int n_elements,
                                const matidx *__restrict__ col_ids,
                                const matidx *__restrict__ row_ids,
                                const data_type *__restrict__ data,
                                const data_type *__restrict__ Y,
                                data_type *__restrict__ Fattr, const int n);

template <typename data_type>
void gpu_hybrid_spmv(int elements_in_rows, int coo_size, data_type *Y,
                     unsigned int rows_count, data_type *F,
                     unsigned int *ell_cols, data_type *ell_data,
                     data_type *coo_data, unsigned int *coo_row_ids,
                     unsigned int *coo_col_ids, int d);

#include "hybridpq.cu"
#endif /* HYBRID_CUH */
