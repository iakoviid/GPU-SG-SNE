void MaxNormalizeDeviceVector(
        thrust::device_vector<float> &d_vector) {
    float max_val = thrust::transform_reduce(d_vector.begin(), d_vector.end(),
            tsnecuda::util::FunctionalAbs(), 0.0f, thrust::maximum<float>());
    thrust::constant_iterator<float> division_iterator(max_val);
    thrust::transform(d_vector.begin(), d_vector.end(), division_iterator,
                      d_vector.begin(), thrust::divides<float>());
}

void SymmetrizeMatrix(cusparseHandle_t &handle,
        thrust::device_vector<float> &d_symmetrized_values,
        thrust::device_vector<int32_t> &d_symmetrized_rowptr,
        thrust::device_vector<int32_t> &d_symmetrized_colind,
        thrust::device_vector<float> &d_values,
        thrust::device_vector<int32_t> &d_indices,
        const float magnitude_factor,
        const int num_points,
        const int num_neighbors)
{

    // Allocate memory
    int32_t *csr_row_ptr_a = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csr_row_ptr_a),
               (num_points+1)*sizeof(int32_t));
    int32_t *csr_column_ptr_a = thrust::raw_pointer_cast(d_indices.data());
    float *csr_values_a = thrust::raw_pointer_cast(d_values.data());

    // Copy the data
    thrust::device_vector<int> d_vector_memory(csr_row_ptr_a,
            csr_row_ptr_a+num_points+1);
    thrust::sequence(d_vector_memory.begin(), d_vector_memory.end(),
                     0, static_cast<int32_t>(num_neighbors));
    thrust::copy(d_vector_memory.begin(), d_vector_memory.end(), csr_row_ptr_a);
    cudaDeviceSynchronize();

    // Initialize the matrix descriptor
    cusparseMatDescr_t matrix_descriptor;
    cusparseCreateMatDescr(&matrix_descriptor);
    cusparseSetMatType(matrix_descriptor, CUSPARSE_MATRIX_TYPE_GENERAL);
    cusparseSetMatIndexBase(matrix_descriptor, CUSPARSE_INDEX_BASE_ZERO);

    // Sort the matrix properly
    size_t permutation_buffer_byte_size = 0;
    void *permutation_buffer = NULL;
    int32_t *permutation = NULL;

    // step 1: Allocate memory buffer
    cusparseXcsrsort_bufferSizeExt(handle, num_points, num_points,
            num_points*num_neighbors, csr_row_ptr_a,
            csr_column_ptr_a, &permutation_buffer_byte_size);
    cudaDeviceSynchronize();
    cudaMalloc(&permutation_buffer,
               sizeof(char)*permutation_buffer_byte_size);

    // step 2: Setup permutation vector permutation to be the identity
    cudaMalloc(reinterpret_cast<void**>(&permutation),
            sizeof(int32_t)*num_points*num_neighbors);
    cusparseCreateIdentityPermutation(handle, num_points*num_neighbors,
                                      permutation);
    cudaDeviceSynchronize();

    // step 3: Sort CSR format
    cusparseXcsrsort(handle, num_points, num_points,
            num_points*num_neighbors, matrix_descriptor, csr_row_ptr_a,
            csr_column_ptr_a, permutation, permutation_buffer);
    cudaDeviceSynchronize();

    // step 4: Gather sorted csr_values
    float* csr_values_a_sorted = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csr_values_a_sorted),
            (num_points*num_neighbors)*sizeof(float));
    cusparseSgthr(handle, num_points*num_neighbors, csr_values_a,
            csr_values_a_sorted, permutation, CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Free some memory
    cudaFree(permutation_buffer);
    cudaFree(permutation);
    csr_values_a = csr_values_a_sorted;

    // We need A^T, so we do a csr2csc() call
    int32_t* csc_row_ptr_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_row_ptr_at),
            (num_points*num_neighbors)*sizeof(int32_t));
    int32_t* csc_column_ptr_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_column_ptr_at),
            (num_points+1)*sizeof(int32_t));
    float* csc_values_at = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&csc_values_at),
            (num_points*num_neighbors)*sizeof(float));

    // Do the transpose operation
    cusparseScsr2csc(handle, num_points, num_points,
                     num_neighbors*num_points, csr_values_a, csr_row_ptr_a,
                     csr_column_ptr_a, csc_values_at, csc_row_ptr_at,
                     csc_column_ptr_at, CUSPARSE_ACTION_NUMERIC,
                     CUSPARSE_INDEX_BASE_ZERO);
    cudaDeviceSynchronize();

    // Now compute the output size of the matrix
    int32_t base_C, num_nonzeros_C;
    int32_t symmetrized_num_nonzeros = -1;
    cusparseSetPointerMode(handle, CUSPARSE_POINTER_MODE_HOST);
    d_symmetrized_rowptr.resize(num_points+1);
    cusparseXcsrgeamNnz(handle, num_points, num_points,
            matrix_descriptor, num_points*num_neighbors, csr_row_ptr_a,
                csr_column_ptr_a,
            matrix_descriptor, num_points*num_neighbors, csc_column_ptr_at,
                csc_row_ptr_at,
            matrix_descriptor,
            thrust::raw_pointer_cast(d_symmetrized_rowptr.data()),
            &symmetrized_num_nonzeros);
    cudaDeviceSynchronize();

    // Do some useful checking...
    if (-1 != symmetrized_num_nonzeros) {
        num_nonzeros_C = symmetrized_num_nonzeros;
    } else {
        cudaMemcpy(&num_nonzeros_C,
                thrust::raw_pointer_cast(d_symmetrized_rowptr.data()) +
                num_points, sizeof(int32_t), cudaMemcpyDeviceToHost);
        cudaMemcpy(&base_C,
                thrust::raw_pointer_cast(d_symmetrized_rowptr.data()),
                sizeof(int), cudaMemcpyDeviceToHost);
    }

    // Allocate memory for the new summed array
    d_symmetrized_colind.resize(num_nonzeros_C);
    d_symmetrized_values.resize(num_nonzeros_C);

    // Sum the arrays
    float kAlpha = 1.0f / (2.0f * num_points);
    float kBeta = 1.0f / (2.0f * num_points);

    cusparseScsrgeam(handle, num_points, num_points,
            &kAlpha, matrix_descriptor, num_points*num_neighbors,
            csr_values_a, csr_row_ptr_a, csr_column_ptr_a,
            &kBeta, matrix_descriptor, num_points*num_neighbors,
            csc_values_at, csc_column_ptr_at, csc_row_ptr_at,
            matrix_descriptor,
            thrust::raw_pointer_cast(d_symmetrized_values.data()),
            thrust::raw_pointer_cast(d_symmetrized_rowptr.data()),
            thrust::raw_pointer_cast(d_symmetrized_colind.data()));
    cudaDeviceSynchronize();

    // Free the memory we were using...
    cudaFree(csr_values_a);
    cudaFree(csc_values_at);
    cudaFree(csr_row_ptr_a);
    cudaFree(csc_column_ptr_at);
    cudaFree(csc_row_ptr_at);
}
