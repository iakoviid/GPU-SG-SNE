void cusparse_bsrmv (
  cusparseHandle_t  &handle,
  cusparseMatDescr_t  &descr_A,
  cusparseDirection_t direction,

  int n_rows,
  int n_cols,
  int nnzb,
  int bs,

  const float *A,
  const int *row_ptr,
  const int *columns,
  const float *x,
  float *y
  )
{
  const float alpha = 1.0;
  const float beta = 0.0;

  cusparseSbsrmv (
    handle,
    direction,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    n_rows, n_cols, nnzb,
    &alpha, descr_A, A,
    row_ptr, columns, bs,
    x, &beta, y);
}

void cusparse_bsrmv (
  cusparseHandle_t  &handle,
  cusparseMatDescr_t  &descr_A,
  cusparseDirection_t direction,

  int n_rows,
  int n_cols,
  int nnzb,
  int bs,

  const double *A,
  const int *row_ptr,
  const int *columns,
  const double *x,
  double *y
)
{
  const double alpha = 1.0;
  const double beta = 0.0;

  cusparseDbsrmv (
    handle,
    direction,
    CUSPARSE_OPERATION_NON_TRANSPOSE,
    n_rows, n_cols, nnzb,
    &alpha, descr_A, A,
    row_ptr, columns, bs,
    x, &beta, y);
}

/// cuSPARSE Column major
{
  cusparseHandle_t handle;
  cusparseCreate (&handle);

  cusparseMatDescr_t descr_A;
  cusparseCreateMatDescr (&descr_A);
  cusparseSetMatType (descr_A, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase (descr_A, CUSPARSE_INDEX_BASE_ZERO);

  cudaEvent_t start, stop;
  cudaEventCreate (&start);
  cudaEventCreate (&stop);

  cudaDeviceSynchronize ();
  cudaEventRecord (start);

  cusparse_bsrmv (handle, descr_A, CUSPARSE_DIRECTION_COLUMN, matrix.n_rows, matrix.n_cols, matrix.nnzb, matrix.bs, d_values, d_row_ptr, d_columns, d_x, d_y);

  cudaEventRecord (stop);
  cudaEventSynchronize (stop);

  float milliseconds = 0;
  cudaEventElapsedTime (&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy (start);
  cudaEventDestroy (stop);

  cusparseDestroyMatDescr (descr_A);
  cusparseDestroy (handle);

  results.emplace_back ("GPU BSR (cuSPARSE, column major)", elapsed, 0, 0);

  cudaMemcpy (cpu_y.get (), d_y, y_size * sizeof (data_type), cudaMemcpyDeviceToHost);
  compare_results (y_size, reference_y, cpu_y.get ());
}
