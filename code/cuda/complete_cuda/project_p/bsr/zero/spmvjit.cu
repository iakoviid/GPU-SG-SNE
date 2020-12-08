#include "spmvjit.cuh"
template <typename data_type, typename index_type>
void
 spmv_jit(bcsr_matrix_class<data_type, index_type> &block_matrix,
              data_type *h_x, data_type *cpu_y) {

  const index_type n_rows = block_matrix.n_rows;
  const index_type bs = block_matrix.bs;
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

    __shared__ float partial_sums[{{ shared_size }}]; // = shared_memory<float> (); ///< Size is equal to blockDim.x * sizeof(float)

    float local_out = 0.0;

    for (; col < last_block * bs; col += 32 / bs)
      {
        const int block = col / bs;
        const int c = col % bs;

        const float value = data[block * bs * bs + c * bs + r];
        const float x_value = x[col_ids[block] * bs + c];
        local_out += x_value * value;
      }

    partial_sums[threadIdx.x] = local_out;

    for (int stride = {{ stride_begin }} ; stride > 0; stride /= 2)
      {
        __syncthreads ();
        if ((lane < stride * bs) && ((threadIdx.x + stride * bs) < 32))
          {
            partial_sums[threadIdx.x] += partial_sums[threadIdx.x + stride * bs];
          }
      }

    if (lane < bs)
      {
        y[block_row * bs + lane] = partial_sums[threadIdx.x];
      }
  },
    (const int *, col_ids),
    (const int *, row_ptr),
    (const float *, data),
    (const float *, x),
    (float*, y));
  nlohmann::json json;
  json["bs"] = bs;
  json["stride_begin"] = round_up_to_power_of_two((32 / bs) / 2);
  json["shared_size"] = block_size.x;
  auto bcsr_kernel = bcsr_jit.compile (json);

  const index_type matrix_size =
      block_matrix.nnzb * block_matrix.bs * block_matrix.bs;
  const index_type columns_size = block_matrix.nnzb;
  const index_type row_ptr_size = block_matrix.n_rows + 1;
  const index_type x_size = block_matrix.n_cols * block_matrix.bs;
  const index_type y_size = block_matrix.n_rows * block_matrix.bs;

  data_type *d_values{};
  data_type *d_y{};
  data_type *d_x{};

  index_type *d_row_ptr{};
  index_type *d_columns{};

  cudaMalloc(&d_values, matrix_size * sizeof(data_type));
  cudaMalloc(&d_x, x_size * sizeof(data_type));
  cudaMalloc(&d_y, y_size * sizeof(data_type));

  cudaMalloc(&d_row_ptr, row_ptr_size * sizeof(index_type));
  cudaMalloc(&d_columns, columns_size * sizeof(index_type));

  cudaMemcpy(d_values, transposed_matrix_data.get(),
             matrix_size * sizeof(data_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_columns, block_matrix.columns.get(),
             columns_size * sizeof(index_type), cudaMemcpyHostToDevice);
  cudaMemcpy(d_row_ptr, block_matrix.row_ptr.get(),
             row_ptr_size * sizeof(index_type), cudaMemcpyHostToDevice);

  cudaMemcpy(d_x, h_x, x_size * sizeof(float), cudaMemcpyHostToDevice);

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaDeviceSynchronize();
  cudaEventRecord(start);

  bcsr_kernel.launch(grid_size, block_size, d_columns, d_row_ptr, d_values, d_x,
                     d_y);

  cudaEventRecord(stop);
  cudaEventSynchronize(stop);

  cudaMemcpy(cpu_y, d_y, y_size * sizeof(data_type), cudaMemcpyDeviceToHost);
  cudaFree(d_values);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_row_ptr);
  cudaFree(d_columns);

  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  const double elapsed = milliseconds / 1000;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  printf("Elapsed jit %lf\n",elapsed );
}
int main(){
return 0}
