// Performs the operation matrix[i, :] = binary_op(matrix[i, :],
// alpha * vector) for each row i in the matrix
template <typename BinaryFunction, typename T>
__global__ void BroadcastRowVector(T *__restrict__ d_matrix,
                                   const T *__restrict__ d_vector, const int N,
                                   const int M, BinaryFunction binary_operation,
                                   const T alpha) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  const int i = TID % N;
  const int j = TID / N;
  if (j < M && d_vector[j]!=0)
    d_matrix[j * N + i] =
        binary_operation(d_matrix[j * N + i], alpha * d_vector[j]);
}

// Performs the operation matrix[:, j] = binary_op(matrix[:, j],
// alpha * vector) for each col i in the matrix
template <typename BinaryFunction, typename T>
__global__ void
BroadcastColumnVector(T *__restrict__ d_matrix, const T *__restrict__ d_vector,
                      const int N, const int M, BinaryFunction binary_operation,
                      const T alpha) {
  const int TID = threadIdx.x + blockIdx.x * blockDim.x;
  const int i = TID % N;
  const int j = TID / N;

  if (j < M && d_vector[i]!=0)
    d_matrix[j * N + i] =
        binary_operation(d_matrix[j * N + i], alpha * d_vector[i]);
}

template <typename BinaryFunction, typename T>
void BroadcastMatrixVector(thrust::device_vector<T> &d_matrix,
                           const thrust::device_vector<T> &d_vector,
                           const int N, const int M,
                           BinaryFunction binary_operation, const int axis,
                           const T alpha) {
  // Checks to make sure dimensions are correct
  assert(d_matrix.size() >= N * M);
  assert((axis == 0 && d_vector.size() >= N) ||
         (axis == 1 && d_vector.size() >= M));

  const int kBlockSize = 32;
  const int kNumBlocks = iDivUp(N * M, kBlockSize);
  if (axis == 0) {
    BroadcastColumnVector<<<kNumBlocks, kBlockSize>>>(
        thrust::raw_pointer_cast(d_matrix.data()),
        thrust::raw_pointer_cast(d_vector.data()), N, M, binary_operation,
        alpha);
  } else {
    BroadcastRowVector<<<kNumBlocks, kBlockSize>>>(
        thrust::raw_pointer_cast(d_matrix.data()),
        thrust::raw_pointer_cast(d_vector.data()), N, M, binary_operation,
        alpha);
  }
}
template void BroadcastMatrixVector<thrust::divides<float>, float>(
        thrust::device_vector<float> &d_matrix,
        const thrust::device_vector<float> &d_vector,
        const int N,
        const int M,
        thrust::divides<float> binary_operation,
        const int axis,
        const float alpha);
// expects matrix of size N x M
thrust::device_vector<float>
ReduceAlpha(cublasHandle_t &handle,
            const thrust::device_vector<float> &d_matrix, const int N,
            const int M, float alpha, const int axis) {
  if (axis == 0) {
    thrust::device_vector<float> ones(N, 1.f);
    thrust::device_vector<float> means(M);

    float kBeta = 0.f;
    cublasSgemv(handle, CUBLAS_OP_T, N, M, &alpha,
                thrust::raw_pointer_cast(d_matrix.data()), N,
                thrust::raw_pointer_cast(ones.data()), 1, &kBeta,
                thrust::raw_pointer_cast(means.data()), 1);
    return means;
  } else if (axis == 1) {
    thrust::device_vector<float> ones(M, 1.f);
    thrust::device_vector<float> means(N);

    float kBeta = 0.f;
    cublasSgemv(handle, CUBLAS_OP_N, N, M, &alpha,
                thrust::raw_pointer_cast(d_matrix.data()), N,
                thrust::raw_pointer_cast(ones.data()), 1, &kBeta,
                thrust::raw_pointer_cast(means.data()), 1);
    return means;
  } else {
    throw std::runtime_error("Axis must be 0 or 1.");
  }
}

thrust::device_vector<float>
ReduceSum(cublasHandle_t &handle, const thrust::device_vector<float> &d_matrix,
          const int N, const int M, const int axis) {
  float alpha = 1.f;
  return ReduceAlpha(handle, d_matrix, N, M, alpha, axis);
}
