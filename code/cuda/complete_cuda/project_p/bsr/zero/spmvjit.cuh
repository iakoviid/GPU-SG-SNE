

#ifndef SPMVJIT_CUH
#define SPMVJIT_CUH
#include "bsr.hpp"
#include <memory>
#include "./include/cuda_jit.h"

template <typename data_type, typename index_type>
void
 spmv_jit(bcsr_matrix_class<data_type, index_type> &block_matrix,
              data_type *h_x, data_type *cpu_y);
#endif
