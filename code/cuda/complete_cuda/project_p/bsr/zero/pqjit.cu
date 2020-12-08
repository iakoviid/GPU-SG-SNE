#include "pqjit.cuh"
template <typename data_type, typename index_type>
void
 pq_jit(bcsr_matrix_class<data_type, index_type> &block_matrix,
              data_type *Y, data_type *Fattr,index_type n, index_type d) {
  const index_type n_rows = block_matrix.n_rows;
  const index_type bs = block_matrix.bs;
  printf("--------------------bs=%d\n",bs );
  dim3 block_size = 32;
  dim3 grid_size{};
  std::unique_ptr<data_type[]> transposed_matrix_data(
      new data_type[block_matrix.size()]);
  block_matrix.transpose_blocks(transposed_matrix_data.get());
  grid_size.x = (block_matrix.n_rows * 32 + block_size.x - 1) / block_size.x;

  jit(bcsr_jit,
  {
    const int bs = {{ bs }};

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = idx % 32;
    const int block_row = idx / 32; ///< Warp per block row
    const int first_block = row_ptr[block_row];
    const int last_block = row_ptr[block_row + 1];

    int col = first_block * bs + lane / bs;
    int r = lane % bs;

    __shared__ coord partial_sums[{{ shared_size }}]; // = shared_memory<float> (); ///< Size is equal to blockDim.x * sizeof(float)

    coord sum1=0;
    coord sum2=0;
    coord sum3=0;

    for (; col < last_block * bs; col += 32 / bs)
      {
        const int block = col / bs;
        const int c = col % bs;
        int column=col_ids[block] * bs + c;
        int row=block_row*bs+r;
        coord dist=0;
        for(int dim=0;dim<d;dim++){
          dist+=(Y[row+dim*n]-Y[column+dim*n])*(Y[row+dim*n]-Y[column+dim*n]);
        }
        const coord value = data[block * bs * bs + c * bs + r]/(1+dist);
        switch (d) {
          case 1:
            sum1+=value*(Y[row]- Y[column]);
            break;
          case 2:
            sum1+=value*(Y[row]-Y[column]);
            sum2+=value*(Y[row+n]-Y[column+n]);
            break;
          case 3:
            sum1+=value*(Y[row]-Y[column]);
            sum2+=value*(Y[row+n]-Y[column+n]);
            sum3+=value*(Y[row+2*n]-Y[column+2*n]);
            break;
        }
      }
      switch (d) {
        case 1:
        partial_sums[threadIdx.x] = sum1;
        break;
        case 2:
        partial_sums[threadIdx.x] = sum1;
        partial_sums[threadIdx.x+blockDim.x]=sum2;
        break;
        case 3:
        partial_sums[threadIdx.x] = sum1;
        partial_sums[threadIdx.x+blockDim.x]=sum2;
        partial_sums[threadIdx.x+2*blockDim.x]=sum3;
        break;

      }

    for (int stride = {{ stride_begin }} ; stride > 0; stride /= 2)
      {
        __syncthreads ();
        if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
          {
            for(int dim=0;dim<d;dim++){
              partial_sums[threadIdx.x+blockDim.x*dim] += partial_sums[threadIdx.x + stride * bs+blockDim.x*dim];

            }

          }
      }

    if (lane < bs)
      {
      for(int dim=0;dim<d;dim++){
        Fattr[block_row * bs + lane+dim*n] = partial_sums[threadIdx.x+blockDim.x*dim];
      }
      }
  },
    (const int *, col_ids),
    (const int *, row_ptr),
    (const coord *, data),
    (const coord *, Y),
    (coord*, Fattr),
    (int , n),
    (int , d)
  );
  nlohmann::json json;
  json["bs"] = bs;
  json["stride_begin"] = round_up_to_power_of_two((32 / bs) / 2);
  json["shared_size"] = block_size.x*d;
  auto bcsr_kernel = bcsr_jit.compile (json);

  const index_type matrix_size =
      block_matrix.nnzb * block_matrix.bs * block_matrix.bs;
  const index_type columns_size = block_matrix.nnzb;
  const index_type row_ptr_size = block_matrix.n_rows + 1;


  data_type *d_values{};


  index_type *d_row_ptr{};
  index_type *d_columns{};

  cudaMalloc(&d_values, matrix_size * sizeof(data_type));


  cudaMalloc(&d_row_ptr, row_ptr_size * sizeof(index_type));
  cudaMalloc(&d_columns, columns_size * sizeof(index_type));

  cudaMemcpy(d_values, transposed_matrix_data.get(),
             matrix_size * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_columns, block_matrix.columns.get(),
             columns_size * sizeof(index_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, block_matrix.row_ptr.get(),
             row_ptr_size * sizeof(index_type), cudaMemcpyHostToDevice);


  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  bcsr_kernel.launch(grid_size, block_size, d_columns, d_row_ptr, d_values, Y,
                     Fattr,n,d);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaFree(d_values);
  cudaFree(d_row_ptr);
  cudaFree(d_columns);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Elapsed jit %lf\n",elapsed );
}
