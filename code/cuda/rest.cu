double computeFrepulsive_interp(double *Ft, double *y, int n, int d, double h)
 {
   thrust::device_vector<float> chargesQij(n*4);
   thrust::device_vector<float> points_device(2*n);
   thrust::host_vector<float> Y(2*n);
   for(int i=0;i<n;i++){
     Y[i]=y[2*i];
     Y[i+n]=y[2*i+1];
   }
   points_device=Y;
   ComputeChargesQij(chargesQij,points_device,n, 4);
   thrust::device_vector<float> repulsive_forces_device;
   thrust::device_vector<float> normalization_vec_device;

   cufftHandle plan;
   float x_max=1;
   float x_min=0;
   float y_max=1;
   float y_min=0;
   thrust::device_vector<float> box_lower_bounds_device;
   thrust::device_vector<float> box_upper_bounds_device;
   thrust::device_vector<float> kernel_tilde_device;
   thrust::device_vector<thrust::complex<float> > fft_kernel_tilde_device;
   PrecomputeFFT2D(plan,x_max,x_min,y_max,y_min,100,3,box_lower_bounds_device,box_upper_bounds_device,kernel_tilde_device,fft_kernel_tilde_device);
   cufftHandle plan_dft;
   cufftHandle plan_idft;
   thrust::device_vector<float> fft_input;
   thrust::device_vector<thrust::complex<float>> fft_w_coefficients;
   thrust::device_vector<float> fft_output;
   thrust::device_vector<int> point_box_idx_device;
   thrust::device_vector<float> x_in_box_device;
   thrust::device_vector<float> y_in_box_device;
   thrust::device_vector<float> y_tilde_spacings_device;
   thrust::device_vector<float> denominator_device;
   thrust::device_vector<float> y_tilde_values;
   thrust::device_vector<float> all_interpolated_values_device;
   thrust::device_vector<float> output_values;
   thrust::device_vector<int> all_interpolated_indices;
   thrust::device_vector<int> output_indices;
   thrust::device_vector<float> w_coefficients_device;
   thrust::device_vector<float> x_interpolated_values_device;
   thrust::device_vector<float> y_interpolated_values_device;
   thrust::device_vector<float> potentialsQij_device;

   NbodyFFT2D(plan_dft,plan_idft,n,4,100,3,fft_kernel_tilde_device,10000,300*300,0,(float)1/100,150,300,fft_input,fft_w_coefficients,
       fft_output,
       point_box_idx_device,
       x_in_box_device,
       y_in_box_device,
       points_device,
       box_lower_bounds_device,
       y_tilde_spacings_device,
       denominator_device,
       y_tilde_values,
       all_interpolated_values_device,
       output_values,
       all_interpolated_indices,
       output_indices,
       w_coefficients_device,
       chargesQij,
       x_interpolated_values_device,
       y_interpolated_values_device,
       potentialsQij_device);


   double z=ComputeRepulsiveForces(repulsive_forces_device,normalization_vec_device,points_device,potentialsQij_device,n,4);
   printf("z=%lf\n",z );
   thrust::host_vector<float> Frep(2*n);
   Frep=repulsive_forces_device;
   for(int i=0;i<n;i++){
     Ft[2*i]=Frep[i];
     Ft[2*i+1]=Frep[i+n];
   }

   return z;
}
__global__ void compute_repulsive_forces_kernel(
    volatile float *__restrict__ repulsive_forces_device,
    volatile float *__restrict__ normalization_vec_device,
    const float *const xs, const float *const ys,
    const float *const potentialsQij, const int num_points, const int n_terms)
    {
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points)
    return;

  register float phi1, phi2, phi3, phi4, x_pt, y_pt;

  phi1 = potentialsQij[TID * n_terms + 0];
  phi2 = potentialsQij[TID * n_terms + 1];
  phi3 = potentialsQij[TID * n_terms + 2];
  phi4 = potentialsQij[TID * n_terms + 3];

  x_pt = xs[TID];
  y_pt = ys[TID];

  normalization_vec_device[TID] = (1 + x_pt * x_pt + y_pt * y_pt) * phi1 -
                                  2 * (x_pt * phi2 + y_pt * phi3) + phi4;

  repulsive_forces_device[TID] = x_pt * phi1 - phi2;
  repulsive_forces_device[TID + num_points] = y_pt * phi1 - phi3;
}
__global__ void ComputeForces(double *Fdx,double* Fdy,double* yd,double* xd,int n,double *PhiScat,int d){
  register float phi1, phi2, phi3, phi4, x, y;

  for(register int tid=threadIdx.x+blockIdx.x*blockDim.x; tid<n ;tid=tid+gridDim.x*blockDim.x){
    phi1 = potentialsQij[tid * 4 + 0];
    phi2 = potentialsQij[tid * 4 + 1];
    phi3 = potentialsQij[tid * 4 + 2];
    phi4 = potentialsQij[tid * 4 + 3];
    x=xd[tid];
    y=y[tid];


  }
}
float zetaAndForce(double *Fdx,double* Fdy,double* yd,double* xd,int n,double *PhiScat,int d){
  int Blocks=64;
  int threads=1024;

  ComputeForces<<<Blocks,threads>>>(Fdx,Fdy,yd,xd,n,PhiScat,d);
  int z = thrust::reduce(Q.begin(),
                              Q.end(), 0.0,
                              thrust::plus<double>());
  return z-n;
}

float ComputeRepulsiveForces(
    thrust::device_vector<float> &repulsive_forces_device,
    thrust::device_vector<float> &normalization_vec_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &potentialsQij, const int num_points,
    const int n_terms)
    {
  const int num_threads = 1024;
  const int num_blocks = (num_points + num_threads - 1) / num_threads;
  compute_repulsive_forces_kernel<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(repulsive_forces_device.data()),
      thrust::raw_pointer_cast(normalization_vec_device.data()),
      thrust::raw_pointer_cast(points_device.data()),
      thrust::raw_pointer_cast(points_device.data() + num_points),
      thrust::raw_pointer_cast(potentialsQij.data()), num_points, n_terms);
  float sumQ = thrust::reduce(normalization_vec_device.begin(),
                              normalization_vec_device.end(), 0.0f,
                              thrust::plus<float>());
  return sumQ - num_points;
}

__global__ void compute_chargesQij_kernel(volatile float *__restrict__ chargesQij,
                          const float *const xs, const float *const ys,
                          const int num_points, const int n_terms)
    {
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_points)
    return;

  register float x_pt, y_pt;
  x_pt = xs[TID];
  y_pt = ys[TID];

  chargesQij[TID * n_terms + 0] = 1;
  chargesQij[TID * n_terms + 1] = x_pt;
  chargesQij[TID * n_terms + 2] = y_pt;
  chargesQij[TID * n_terms + 3] = x_pt * x_pt + y_pt * y_pt;
}

void ComputeChargesQij(thrust::device_vector<float> &chargesQij,
                       thrust::device_vector<float> &points_device,
                       const int num_points, const int n_terms)
    {
  const int num_threads = 1024;
  const int num_blocks = (num_points + num_threads - 1) / num_threads;
  compute_chargesQij_kernel<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(chargesQij.data()),
      thrust::raw_pointer_cast(points_device.data()),
      thrust::raw_pointer_cast(points_device.data() + num_points), num_points,
      n_terms);
}
__global__ void chargesQijKernel(double* xd,double* yd,double* chargesQij,int n)
{
  register int  n_terms=4;
  for(register int tid=threadIdx.x+blockIdx.x*blockDim.x;tid<n;tid=tid+gridDim.x*blockDim.x )
  {
    register double x,y;
    x=xd[tid];
    y=yd[tid];
    chargesQij[tid * n_terms + 0] = 1;
    chargesQij[tid * n_terms + 1] = x;
    chargesQij[tid * n_terms + 2] = y;
    chargesQij[tid * n_terms + 3] = x * x + y * y;
}

}
void ComputeChargesQijmine(double* xd,double* yd,double* chargesQij,int n,int d){
  int  threads=1024;
  int Blocks=64;
  chargesQijKernel<<<Blocks,threads>>>(xd,yd,chargesQij,n);
}


__global__ void copy_to_fft_input(volatile float * __restrict__ fft_input,
                                  const float * w_coefficients_device,
                                  const int n_fft_coeffs,
                                  const int n_fft_coeffs_half,
                                  const int n_terms)
{
    register int i, j;
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    register int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
    register int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half;

    fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] = w_coefficients_device[current_term + current_loc * n_terms];
}

__global__ void copy_from_fft_output(volatile float * __restrict__ y_tilde_values,
    const float * fft_output,
    const int n_fft_coeffs,
    const int n_fft_coeffs_half,
    const int n_terms)
{
    register int i, j;
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
        return;

    register int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
    register int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

    i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
    j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

    y_tilde_values[current_term + n_terms * current_loc] = fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs + j] / (float) (n_fft_coeffs * n_fft_coeffs);
}

__global__ void compute_point_box_idx(volatile int * __restrict__ point_box_idx,
                                      volatile float * __restrict__ x_in_box,
                                      volatile float * __restrict__ y_in_box,
                                      const float * const xs,
                                      const float * const ys,
                                      const float * const box_lower_bounds,
                                      const float coord_min,
                                      const float box_width,
                                      const int n_boxes,
                                      const int n_total_boxes,
                                      const int N)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N)
        return;

    register int x_idx = (int) ((xs[TID] - coord_min) / box_width);
    register int y_idx = (int) ((ys[TID] - coord_min) / box_width);

    x_idx = max(0, x_idx);
    x_idx = min(n_boxes - 1, x_idx);

    y_idx = max(0, y_idx);
    y_idx = min(n_boxes - 1, y_idx);

    register int box_idx = y_idx * n_boxes + x_idx;
    point_box_idx[TID] = box_idx;

    x_in_box[TID] = (xs[TID] - box_lower_bounds[box_idx]) / box_width;
    y_in_box[TID] = (ys[TID] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

__global__ void interpolate_device(
    volatile float * __restrict__ interpolated_values,
    const float * const y_in_box,
    const float * const y_tilde_spacings,
    const float * const denominator,
    const int n_interpolation_points,
    const int N)
{
    register int TID, i, j, k;
    register float value, ybox_i;

    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= N * n_interpolation_points)
        return;

    i = TID % N;
    j = TID / N;

    value = 1;
    ybox_i = y_in_box[i];

    for (k = 0; k < n_interpolation_points; k++) {
        if (j != k) {
            value *= ybox_i - y_tilde_spacings[k];
        }
    }

    interpolated_values[j * N + i] = value / denominator[j];
}

__global__ void compute_interpolated_indices(
    float * __restrict__ w_coefficients_device,
    const int * const point_box_indices,
    const float * const chargesQij,
    const float * const x_interpolated_values,
    const float * const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms)
{
    register int TID, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = TID % n_terms;
    i = (TID / n_terms) % N;
    interp_j = ((TID / n_terms) / N) % n_interpolation_points;
    interp_i = ((TID / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * chargesQij[i * n_terms + current_term];
    idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                                (box_j * n_interpolation_points) + interp_j;
    // interpolated_indices[TID] = idx * n_terms + current_term;
    atomicAdd(
        w_coefficients_device + idx * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * chargesQij[i * n_terms + current_term]);
}

__global__ void compute_potential_indices(
    float * __restrict__ potentialsQij,
    const int * const point_box_indices,
    const float * const y_tilde_values,
    const float * const x_interpolated_values,
    const float * const y_interpolated_values,
    const int N,
    const int n_interpolation_points,
    const int n_boxes,
    const int n_terms)
{
    register int TID, current_term, i, interp_i, interp_j, box_idx, box_i, box_j, idx;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N)
        return;

    current_term = TID % n_terms;
    i = (TID / n_terms) % N;
    interp_j = ((TID / n_terms) / N) % n_interpolation_points;
    interp_i = ((TID / n_terms) / N) / n_interpolation_points;

    box_idx = point_box_indices[i];
    box_i = box_idx % n_boxes;
    box_j = box_idx / n_boxes;

    idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                                (box_j * n_interpolation_points) + interp_j;
    // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term];
    // interpolated_indices[TID] = i * n_terms + current_term;
    atomicAdd(
        potentialsQij + i * n_terms + current_term,
        x_interpolated_values[i + interp_i * N] * y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms + current_term]);
}

__host__ __device__ float squared_cauchy_2d(float x1, float x2, float y1, float y2) {
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}

__global__ void compute_kernel_tilde(
    volatile float * __restrict__ kernel_tilde,
    const float x_min,
    const float y_min,
    const float h,
    const int n_interpolation_points_1d,
    const int n_fft_coeffs)
{
    register int TID, i, j;
    register float tmp;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_interpolation_points_1d * n_interpolation_points_1d)
        return;

    i = TID / n_interpolation_points_1d;
    j = TID % n_interpolation_points_1d;

    tmp = squared_cauchy_2d(y_min + h / 2, x_min + h / 2, y_min + h / 2 + i * h, x_min + h / 2 + j * h);
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
    kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;

}

__global__ void compute_upper_and_lower_bounds(
    volatile float * __restrict__ box_upper_bounds,
    volatile float * __restrict__ box_lower_bounds,
    const float box_width,
    const float x_min,
    const float y_min,
    const int n_boxes,
    const int n_total_boxes)
{
    register int TID, i, j;
    TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= n_boxes * n_boxes)
        return;

    i = TID / n_boxes;
    j = TID % n_boxes;

    box_lower_bounds[i * n_boxes + j] = j * box_width + x_min;
    box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

    box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
    box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
}

__global__ void copy_to_w_coefficients(
    volatile float * __restrict__ w_coefficients_device,
    const int * const output_indices,
    const float * const output_values,
    const int num_elements)
{
    register int TID = threadIdx.x + blockIdx.x * blockDim.x;
    if (TID >= num_elements)
        return;

    w_coefficients_device[output_indices[TID]] = output_values[TID];
}

void PrecomputeFFT2D(
        cufftHandle &plan_kernel_tilde,
        float x_max,
        float x_min,
        float y_max,
        float y_min,
        int n_boxes,
        int n_interpolation_points,
        thrust::device_vector<float> &box_lower_bounds_device,
        thrust::device_vector<float> &box_upper_bounds_device,
        thrust::device_vector<float> &kernel_tilde_device,
        thrust::device_vector<thrust::complex<float> > &fft_kernel_tilde_device)
        {
    const int num_threads = 32;
    int num_blocks = (n_boxes * n_boxes + num_threads - 1) / num_threads;
    /*
     * Set up the boxes
     */
    int n_total_boxes = n_boxes * n_boxes;
    float box_width = (x_max - x_min) / (float) n_boxes;

    // Left and right bounds of each box, first the lower bounds in the x direction, then in the y direction
    compute_upper_and_lower_bounds<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(box_upper_bounds_device.data()),
        thrust::raw_pointer_cast(box_lower_bounds_device.data()),
        box_width, x_min, y_min, n_boxes, n_total_boxes);

    // Coordinates of all the equispaced interpolation points
    int n_interpolation_points_1d = n_interpolation_points * n_boxes;
    int n_fft_coeffs = 2 * n_interpolation_points_1d;

    float h = box_width / (float) n_interpolation_points;

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    // thrust::device_vector<float> kernel_tilde_device(n_fft_coeffs * n_fft_coeffs);
    num_blocks = (n_interpolation_points_1d * n_interpolation_points_1d + num_threads - 1) / num_threads;
    compute_kernel_tilde<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(kernel_tilde_device.data()),
        x_min, y_min, h, n_interpolation_points_1d, n_fft_coeffs);
    cudaDeviceSynchronize();

    // Precompute the FFT of the kernel generating matrix

    cufftExecR2C(plan_kernel_tilde,
        reinterpret_cast<cufftReal *>(thrust::raw_pointer_cast(kernel_tilde_device.data())),
        reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(fft_kernel_tilde_device.data())));

}



void NbodyFFT2D(
    cufftHandle &plan_dft,
    cufftHandle &plan_idft,
    int N,
    int n_terms,
    int n_boxes,
    int n_interpolation_points,
    thrust::device_vector<thrust::complex<float>> &fft_kernel_tilde_device,
    int n_total_boxes,
    int total_interpolation_points,
    float coord_min,
    float box_width,
    int n_fft_coeffs_half,
    int n_fft_coeffs,
    thrust::device_vector<float> &fft_input,
    thrust::device_vector<thrust::complex<float>> &fft_w_coefficients,
    thrust::device_vector<float> &fft_output,
    thrust::device_vector<int> &point_box_idx_device,
    thrust::device_vector<float> &x_in_box_device,
    thrust::device_vector<float> &y_in_box_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &box_lower_bounds_device,
    thrust::device_vector<float> &y_tilde_spacings_device,
    thrust::device_vector<float> &denominator_device,
    thrust::device_vector<float> &y_tilde_values,
    thrust::device_vector<float> &all_interpolated_values_device,
    thrust::device_vector<float> &output_values,
    thrust::device_vector<int> &all_interpolated_indices,
    thrust::device_vector<int> &output_indices,
    thrust::device_vector<float> &w_coefficients_device,
    thrust::device_vector<float> &chargesQij_device,
    thrust::device_vector<float> &x_interpolated_values_device,
    thrust::device_vector<float> &y_interpolated_values_device,
    thrust::device_vector<float> &potentialsQij_device)
     {
    // std::cout << "start" << std::endl;
    const int num_threads = 128;
    int num_blocks = (N + num_threads - 1) / num_threads;

     // Compute box indices and the relative position of each point in its box in the interval [0, 1]
    compute_point_box_idx<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(x_in_box_device.data()),
        thrust::raw_pointer_cast(y_in_box_device.data()),
        thrust::raw_pointer_cast(points_device.data()),
        thrust::raw_pointer_cast(points_device.data() + N),
        thrust::raw_pointer_cast(box_lower_bounds_device.data()),
        coord_min,
        box_width,
        n_boxes,
        n_total_boxes,
        N
    );

    cudaDeviceSynchronize();

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
    num_blocks = (N * n_interpolation_points + num_threads - 1) / num_threads;
    interpolate_device<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(x_in_box_device.data()),
        thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
        thrust::raw_pointer_cast(denominator_device.data()),
        n_interpolation_points,
        N
    );
    cudaDeviceSynchronize();

    // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
    interpolate_device<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_in_box_device.data()),
        thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
        thrust::raw_pointer_cast(denominator_device.data()),
        n_interpolation_points,
        N
    );
    cudaDeviceSynchronize();

    num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N + num_threads - 1) / num_threads;
    compute_interpolated_indices<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(w_coefficients_device.data()),
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(chargesQij_device.data()),
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        N,
        n_interpolation_points,
        n_boxes,
        n_terms
    );
    cudaDeviceSynchronize();

    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */

    num_blocks = ((n_terms * n_fft_coeffs_half * n_fft_coeffs_half) + num_threads - 1) / num_threads;
    copy_to_fft_input<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(fft_input.data()),
        thrust::raw_pointer_cast(w_coefficients_device.data()),
        n_fft_coeffs,
        n_fft_coeffs_half,
        n_terms
    );
    cudaDeviceSynchronize();
    // Compute fft values at interpolated nodes
    cufftExecR2C(plan_dft,
        reinterpret_cast<cufftReal *>(thrust::raw_pointer_cast(fft_input.data())),
        reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(fft_w_coefficients.data())));
    cudaDeviceSynchronize();

    // Take the broadcasted Hadamard product of a complex matrix and a complex vector
    //BroadcastMatrixVector(
    //    fft_w_coefficients, fft_kernel_tilde_device, n_fft_coeffs * (n_fft_coeffs / 2 + 1), n_terms,
  //      thrust::multiplies<thrust::complex<float>>(), 0, thrust::complex<float>(1.0));



    // Invert the computed values at the interpolated nodes
    cufftExecC2R(plan_idft,
        reinterpret_cast<cufftComplex *>(thrust::raw_pointer_cast(fft_w_coefficients.data())),
        reinterpret_cast<cufftReal *>(thrust::raw_pointer_cast(fft_output.data())));
    cudaDeviceSynchronize();
    copy_from_fft_output<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(y_tilde_values.data()),
        thrust::raw_pointer_cast(fft_output.data()),
        n_fft_coeffs,
        n_fft_coeffs_half,
        n_terms
    );
    cudaDeviceSynchronize();

    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
    num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N + num_threads - 1) / num_threads;
    compute_potential_indices<<<num_blocks, num_threads>>>(
        thrust::raw_pointer_cast(potentialsQij_device.data()),
        thrust::raw_pointer_cast(point_box_idx_device.data()),
        thrust::raw_pointer_cast(y_tilde_values.data()),
        thrust::raw_pointer_cast(x_interpolated_values_device.data()),
        thrust::raw_pointer_cast(y_interpolated_values_device.data()),
        N,
        n_interpolation_points,
        n_boxes,
        n_terms
    );
    cudaDeviceSynchronize(  );
}



__global__ void copy_to_fft_input(volatile float *__restrict__ fft_input,
                                  const float *w_coefficients_device,
                                  const int n_fft_coeffs,
                                  const int n_fft_coeffs_half,
                                  const int n_terms)
                                  {
  register int i, j;
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
    return;

  register int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
  register int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

  i = current_loc / n_fft_coeffs_half;
  j = current_loc % n_fft_coeffs_half;

  fft_input[current_term * (n_fft_coeffs * n_fft_coeffs) + i * n_fft_coeffs +
            j] = w_coefficients_device[current_term + current_loc * n_terms];
}

__global__ void copy_from_fft_output(volatile float *__restrict__ y_tilde_values,
                     const float *fft_output, const int n_fft_coeffs,
                     const int n_fft_coeffs_half, const int n_terms) {
  register int i, j;
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_fft_coeffs_half * n_fft_coeffs_half)
    return;

  register int current_term = TID / (n_fft_coeffs_half * n_fft_coeffs_half);
  register int current_loc = TID % (n_fft_coeffs_half * n_fft_coeffs_half);

  i = current_loc / n_fft_coeffs_half + n_fft_coeffs_half;
  j = current_loc % n_fft_coeffs_half + n_fft_coeffs_half;

  y_tilde_values[current_term + n_terms * current_loc] =
      fft_output[current_term * (n_fft_coeffs * n_fft_coeffs) +
                 i * n_fft_coeffs + j] /
      (float)(n_fft_coeffs * n_fft_coeffs);
}

__global__ void compute_point_box_idx(volatile int *__restrict__ point_box_idx,
                                      volatile float *__restrict__ x_in_box,
                                      volatile float *__restrict__ y_in_box,
                                      const float *const xs,
                                      const float *const ys,
                                      const float *const box_lower_bounds,
                                      const float coord_min,
                                      const float box_width, const int n_boxes,
                                      const int n_total_boxes, const int N)
                                      {
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= N)
    return;

  register int x_idx = (int)((xs[TID] - coord_min) / box_width);
  register int y_idx = (int)((ys[TID] - coord_min) / box_width);

  x_idx = max(0, x_idx);
  x_idx = min(n_boxes - 1, x_idx);

  y_idx = max(0, y_idx);
  y_idx = min(n_boxes - 1, y_idx);

  register int box_idx = y_idx * n_boxes + x_idx;
  point_box_idx[TID] = box_idx;

  x_in_box[TID] = (xs[TID] - box_lower_bounds[box_idx]) / box_width;
  y_in_box[TID] =
      (ys[TID] - box_lower_bounds[n_total_boxes + box_idx]) / box_width;
}

__global__ void interpolate_device(volatile float *__restrict__ interpolated_values,
                   const float *const y_in_box,
                   const float *const y_tilde_spacings,
                   const float *const denominator,
                   const int n_interpolation_points, const int N)
                   {
  register int TID, i, j, k;
  register float value, ybox_i;

  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= N * n_interpolation_points)
    return;

  i = TID % N;
  j = TID / N;

  value = 1;
  ybox_i = y_in_box[i];

  for (k = 0; k < n_interpolation_points; k++) {
    if (j != k) {
      value *= ybox_i - y_tilde_spacings[k];
    }
  }

  interpolated_values[j * N + i] = value / denominator[j];
}

__global__ void compute_interpolated_indices(
    float *__restrict__ w_coefficients_device,
    const int *const point_box_indices, const float *const chargesQij,
    const float *const x_interpolated_values,
    const float *const y_interpolated_values, const int N,
    const int n_interpolation_points, const int n_boxes, const int n_terms)
    {
  register int TID, current_term, i, interp_i, interp_j, box_idx, box_i, box_j,
      idx;
  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N)
    return;

  current_term = TID % n_terms;
  i = (TID / n_terms) % N;
  interp_j = ((TID / n_terms) / N) % n_interpolation_points;
  interp_i = ((TID / n_terms) / N) / n_interpolation_points;

  box_idx = point_box_indices[i];
  box_i = box_idx % n_boxes;
  box_j = box_idx / n_boxes;

  // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] *
  // y_interpolated_values[i + interp_j * N] * chargesQij[i * n_terms +
  // current_term];
  idx = (box_i * n_interpolation_points + interp_i) *
            (n_boxes * n_interpolation_points) +
        (box_j * n_interpolation_points) + interp_j;
  // interpolated_indices[TID] = idx * n_terms + current_term;
  atomicAdd(w_coefficients_device + idx * n_terms + current_term,
            x_interpolated_values[i + interp_i * N] *
                y_interpolated_values[i + interp_j * N] *
                chargesQij[i * n_terms + current_term]);
}

__global__ void compute_potential_indices(
    float *__restrict__ potentialsQij, const int *const point_box_indices,
    const float *const y_tilde_values, const float *const x_interpolated_values,
    const float *const y_interpolated_values, const int N,
    const int n_interpolation_points, const int n_boxes, const int n_terms)
     {
  register int TID, current_term, i, interp_i, interp_j, box_idx, box_i, box_j,
      idx;
  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_terms * n_interpolation_points * n_interpolation_points * N)
    return;

  current_term = TID % n_terms;
  i = (TID / n_terms) % N;
  interp_j = ((TID / n_terms) / N) % n_interpolation_points;
  interp_i = ((TID / n_terms) / N) / n_interpolation_points;

  box_idx = point_box_indices[i];
  box_i = box_idx % n_boxes;
  box_j = box_idx / n_boxes;

  idx = (box_i * n_interpolation_points + interp_i) *
            (n_boxes * n_interpolation_points) +
        (box_j * n_interpolation_points) + interp_j;
  // interpolated_values[TID] = x_interpolated_values[i + interp_i * N] *
  // y_interpolated_values[i + interp_j * N] * y_tilde_values[idx * n_terms +
  // current_term]; interpolated_indices[TID] = i * n_terms + current_term;
  atomicAdd(potentialsQij + i * n_terms + current_term,
            x_interpolated_values[i + interp_i * N] *
                y_interpolated_values[i + interp_j * N] *
                y_tilde_values[idx * n_terms + current_term]);
}

__host__ __device__ float squared_cauchy_2d(float x1, float x2, float y1,
                                            float y2) {
  return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -2);
}

__global__ void compute_kernel_tilde(volatile float *__restrict__ kernel_tilde,
                                     const float x_min, const float y_min,
                                     const float h,
                                     const int n_interpolation_points_1d,
                                     const int n_fft_coeffs)
                                     {
  register int TID, i, j;
  register float tmp;
  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_interpolation_points_1d * n_interpolation_points_1d)
    return;

  i = TID / n_interpolation_points_1d;
  j = TID % n_interpolation_points_1d;

  tmp = squared_cauchy_2d(y_min + h / 2, x_min + h / 2, y_min + h / 2 + i * h,
                          x_min + h / 2 + j * h);
  kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs +
               (n_interpolation_points_1d + j)] = tmp;
  kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs +
               (n_interpolation_points_1d + j)] = tmp;
  kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs +
               (n_interpolation_points_1d - j)] = tmp;
  kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs +
               (n_interpolation_points_1d - j)] = tmp;
}

__global__ void compute_upper_and_lower_bounds(volatile float *__restrict__ box_upper_bounds,
                               volatile float *__restrict__ box_lower_bounds,
                               const float box_width, const float x_min,
                               const float y_min, const int n_boxes,
                               const int n_total_boxes)
    {
  register int TID, i, j;
  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= n_boxes * n_boxes)
    return;

  i = TID / n_boxes;
  j = TID % n_boxes;

  box_lower_bounds[i * n_boxes + j] = j * box_width + x_min;
  box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

  box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
  box_upper_bounds[n_total_boxes + i * n_boxes + j] =
      (i + 1) * box_width + y_min;
}

__global__ void
copy_to_w_coefficients(volatile float *__restrict__ w_coefficients_device,
                       const int *const output_indices,
                       const float *const output_values,
                       const int num_elements)
                       {
  register int TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_elements)
    return;

  w_coefficients_device[output_indices[TID]] = output_values[TID];
}

void PrecomputeFFT2D(
    cufftHandle &plan_kernel_tilde, float x_max, float x_min, float y_max,
    float y_min, int n_boxes, int n_interpolation_points,
    thrust::device_vector<float> &box_lower_bounds_device,
    thrust::device_vector<float> &box_upper_bounds_device,
    thrust::device_vector<float> &kernel_tilde_device,
    thrust::device_vector<thrust::complex<float>> &fft_kernel_tilde_device)
    {
  const int num_threads = 32;
  int num_blocks = (n_boxes * n_boxes + num_threads - 1) / num_threads;
  /*
   * Set up the boxes
   */
  int n_total_boxes = n_boxes * n_boxes;
  float box_width = (x_max - x_min) / (float)n_boxes;

  // Left and right bounds of each box, first the lower bounds in the x
  // direction, then in the y direction
  compute_upper_and_lower_bounds<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(box_upper_bounds_device.data()),
      thrust::raw_pointer_cast(box_lower_bounds_device.data()), box_width,
      x_min, y_min, n_boxes, n_total_boxes);

  // Coordinates of all the equispaced interpolation points
  int n_interpolation_points_1d = n_interpolation_points * n_boxes;
  int n_fft_coeffs = 2 * n_interpolation_points_1d;

  float h = box_width / (float)n_interpolation_points;

  /*
   * Evaluate the kernel at the interpolation nodes and form the embedded
   * generating kernel vector for a circulant matrix
   */
  // thrust::device_vector<float> kernel_tilde_device(n_fft_coeffs *
  // n_fft_coeffs);
  num_blocks = (n_interpolation_points_1d * n_interpolation_points_1d +
                num_threads - 1) /
               num_threads;
  compute_kernel_tilde<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(kernel_tilde_device.data()), x_min, y_min, h,
      n_interpolation_points_1d, n_fft_coeffs);
  cudaDeviceSynchronize();

  // Precompute the FFT of the kernel generating matrix

  cufftExecR2C(plan_kernel_tilde,
               reinterpret_cast<cufftReal *>(
                   thrust::raw_pointer_cast(kernel_tilde_device.data())),
               reinterpret_cast<cufftComplex *>(
                   thrust::raw_pointer_cast(fft_kernel_tilde_device.data())));
}

void NbodyFFT2D(
    cufftHandle &plan_dft, cufftHandle &plan_idft, int N, int n_terms,
    int n_boxes, int n_interpolation_points,
    thrust::device_vector<thrust::complex<float>> &fft_kernel_tilde_device,
    int n_total_boxes, int total_interpolation_points, float coord_min,
    float box_width, int n_fft_coeffs_half, int n_fft_coeffs,
    thrust::device_vector<float> &fft_input,
    thrust::device_vector<thrust::complex<float>> &fft_w_coefficients,
    thrust::device_vector<float> &fft_output,
    thrust::device_vector<int> &point_box_idx_device,
    thrust::device_vector<float> &x_in_box_device,
    thrust::device_vector<float> &y_in_box_device,
    thrust::device_vector<float> &points_device,
    thrust::device_vector<float> &box_lower_bounds_device,
    thrust::device_vector<float> &y_tilde_spacings_device,
    thrust::device_vector<float> &denominator_device,
    thrust::device_vector<float> &y_tilde_values,
    thrust::device_vector<float> &all_interpolated_values_device,
    thrust::device_vector<float> &output_values,
    thrust::device_vector<int> &all_interpolated_indices,
    thrust::device_vector<int> &output_indices,
    thrust::device_vector<float> &w_coefficients_device,
    thrust::device_vector<float> &chargesQij_device,
    thrust::device_vector<float> &x_interpolated_values_device,
    thrust::device_vector<float> &y_interpolated_values_device,
    thrust::device_vector<float> &potentialsQij_device)
     {
  // std::cout << "start" << std::endl;
  const int num_threads = 128;
  int num_blocks = (N + num_threads - 1) / num_threads;

  // Compute box indices and the relative position of each point in its box in
  // the interval [0, 1]
  compute_point_box_idx<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(point_box_idx_device.data()),
      thrust::raw_pointer_cast(x_in_box_device.data()),
      thrust::raw_pointer_cast(y_in_box_device.data()),
      thrust::raw_pointer_cast(points_device.data()),
      thrust::raw_pointer_cast(points_device.data() + N),
      thrust::raw_pointer_cast(box_lower_bounds_device.data()), coord_min,
      box_width, n_boxes, n_total_boxes, N);

  cudaDeviceSynchronize();

  /*
   * Step 1: Interpolate kernel using Lagrange polynomials and compute the w
   * coefficients
   */
  // Compute the interpolated values at each real point with each Lagrange
  // polynomial in the `x` direction
  num_blocks = (N * n_interpolation_points + num_threads - 1) / num_threads;
  interpolate_device<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(x_interpolated_values_device.data()),
      thrust::raw_pointer_cast(x_in_box_device.data()),
      thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
      thrust::raw_pointer_cast(denominator_device.data()),
      n_interpolation_points, N);
  cudaDeviceSynchronize();

  // Compute the interpolated values at each real point with each Lagrange
  // polynomial in the `y` direction
  interpolate_device<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(y_interpolated_values_device.data()),
      thrust::raw_pointer_cast(y_in_box_device.data()),
      thrust::raw_pointer_cast(y_tilde_spacings_device.data()),
      thrust::raw_pointer_cast(denominator_device.data()),
      n_interpolation_points, N);
  cudaDeviceSynchronize();

  num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N +
                num_threads - 1) /
               num_threads;
  compute_interpolated_indices<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(w_coefficients_device.data()),
      thrust::raw_pointer_cast(point_box_idx_device.data()),
      thrust::raw_pointer_cast(chargesQij_device.data()),
      thrust::raw_pointer_cast(x_interpolated_values_device.data()),
      thrust::raw_pointer_cast(y_interpolated_values_device.data()), N,
      n_interpolation_points, n_boxes, n_terms);
  cudaDeviceSynchronize();

  /*
   * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the
   * kernel matrix with the coefficients w
   */

  num_blocks =
      ((n_terms * n_fft_coeffs_half * n_fft_coeffs_half) + num_threads - 1) /
      num_threads;
  copy_to_fft_input<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(fft_input.data()),
      thrust::raw_pointer_cast(w_coefficients_device.data()), n_fft_coeffs,
      n_fft_coeffs_half, n_terms);
  cudaDeviceSynchronize();
  // Compute fft values at interpolated nodes
  cufftExecR2C(
      plan_dft,
      reinterpret_cast<cufftReal *>(thrust::raw_pointer_cast(fft_input.data())),
      reinterpret_cast<cufftComplex *>(
          thrust::raw_pointer_cast(fft_w_coefficients.data())));
  cudaDeviceSynchronize();

  // Take the broadcasted Hadamard product of a complex matrix and a complex
  // vector
  tsnecuda::util::BroadcastMatrixVector(
      fft_w_coefficients, fft_kernel_tilde_device,
      n_fft_coeffs * (n_fft_coeffs / 2 + 1), n_terms,
      thrust::multiplies<thrust::complex<float>>(), 0,
      thrust::complex<float>(1.0));

  // Invert the computed values at the interpolated nodes
  cufftExecC2R(plan_idft,
               reinterpret_cast<cufftComplex *>(
                   thrust::raw_pointer_cast(fft_w_coefficients.data())),
               reinterpret_cast<cufftReal *>(
                   thrust::raw_pointer_cast(fft_output.data())));
  cudaDeviceSynchronize();
  copy_from_fft_output<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(y_tilde_values.data()),
      thrust::raw_pointer_cast(fft_output.data()), n_fft_coeffs,
      n_fft_coeffs_half, n_terms);
  cudaDeviceSynchronize();

  /*
   * Step 3: Compute the potentials \tilde{\phi}
   */
  num_blocks = (n_terms * n_interpolation_points * n_interpolation_points * N +
                num_threads - 1) /
               num_threads;
  compute_potential_indices<<<num_blocks, num_threads>>>(
      thrust::raw_pointer_cast(potentialsQij_device.data()),
      thrust::raw_pointer_cast(point_box_idx_device.data()),
      thrust::raw_pointer_cast(y_tilde_values.data()),
      thrust::raw_pointer_cast(x_interpolated_values_device.data()),
      thrust::raw_pointer_cast(y_interpolated_values_device.data()), N,
      n_interpolation_points, n_boxes, n_terms);
  cudaDeviceSynchronize();
}
