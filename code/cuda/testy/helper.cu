__host__ __device__ double kernel1d(double hsq, double i) {
  return pow(1.0 + hsq * i*i, -2);
}

__host__ __device__ double kernel2d(double hsq, double i, double j) {
  return pow(1.0 + hsq * ( i*i + j*j ), -2);
}

__host__ __device__ double kernel3d(double hsq, double i, double j, double k) {
  return pow(1.0 + hsq * ( i*i + j*j + k*k ), -2);}
