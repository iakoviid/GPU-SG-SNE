/*
    -- MAGMA (version 2.0) --
       Univ. of Tennessee, Knoxville
       Univ. of California, Berkeley
       Univ. of Colorado, Denver
       @date

       @precisions normal z -> c d s

*/
#define BLOCK_SIZE 512

// ELLPACK SpMV kernel
//Michael Garland
template <typename data_type, typename index_type>
__global__ void
zgeellmv_kernel(
    index_type num_rows,
    index_type num_cols,
    index_type num_cols_per_row,
    data_type alpha,
    data_type * dval,
    data_type * dcolind,
    data_type * dx,
    data_type beta,
    data_type * dy)
{
    index_type row = blockDim.x * blockIdx.x + threadIdx.x;
    if (row < num_rows) {
        data_type dot = 0;
        for ( index_type n = 0; n < num_cols_per_row; n++ ) {
            index_type col = dcolind [ num_cols_per_row * row + n ];
            data_type val = dval [ num_cols_per_row * row + n ];
            if ( val != 0)
                dot += val * dx[col ];
        }
        dy[ row ] = dot * alpha + beta * dy [ row ];
    }
}



/**
    Purpose
    -------

    This routine computes y = alpha *  A *  x + beta * y on the GPU.
    Input format is ELLPACK.

    Arguments
    ---------

    @param[in]
    transA      magma_trans_t
                transposition parameter for A

    @param[in]
    m           magma_int_t
                number of rows in A

    @param[in]
    n           magma_int_t
                number of columns in A

    @param[in]
    nnz_per_row magma_int_t
                number of elements in the longest row

    @param[in]
    alpha       magmaDoubleComplex
                scalar multiplier

    @param[in]
    dval        magmaDoubleComplex_ptr
                array containing values of A in ELLPACK

    @param[in]
    dcolind     magmaIndex_ptr
                columnindices of A in ELLPACK

    @param[in]
    dx          magmaDoubleComplex_ptr
                input vector x

    @param[in]
    beta        magmaDoubleComplex
                scalar multiplier

    @param[out]
    dy          magmaDoubleComplex_ptr
                input/output vector y

    @param[in]
    queue       magma_queue_t
                Queue to execute in.

    @ingroup magmasparse_zblas
    ********************************************************************/
template <typename data_type, typename index_type>
void magma_zgeellmv(
    index_type m, index_type n,
    index_type nnz_per_row,
    data_type alpha,
    data_typ* dval,
    index_type* dcolind,
    data_type* dx,
    data_type beta,
    data_type* dy )
{
    dim3 grid( ceil( m, BLOCK_SIZE ) );
    index_type threads = BLOCK_SIZE;
    zgeellmv_kernel<<< grid, threads, 0>>>
                  ( m, n, nnz_per_row, alpha, dval, dcolind, dx, beta, dy );


}
