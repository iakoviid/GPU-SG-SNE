#ifndef KERNELS
#define KERNELS
__host__ __device__  double kernel1d(double hsq, double i) ;

__host__ __device__  double kernel2d(double hsq, double i, double j) ;

__host__ __device__  double kernel3d(double hsq, double i, double j, double k);
#endif
