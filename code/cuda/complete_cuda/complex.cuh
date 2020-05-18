#ifndef COMPLEX_CUH
#define COMPLEX_CUH
#include "common.cuh"
typedef float2 Complex;

// Complex addition
static __device__ __host__ inline Complex ComplexAdd(Complex a, Complex b) {
  Complex c;
  c.x = a.x + b.x;
  c.y = a.y + b.y;
  return c;
}

// Complex scale
static __device__ __host__ inline Complex ComplexScale(Complex a, float s) {
  Complex c;
  c.x = s * a.x;
  c.y = s * a.y;
  return c;
}

// Complex multiplication
static __device__ __host__ inline Complex ComplexMul(Complex a, Complex b) {
  Complex c;
  c.x = a.x * b.x - a.y * b.y;
  c.y = a.x * b.y + a.y * b.x;
  return c;
}
__device__ __forceinline__ Complex my_cexpf(Complex z) {

  Complex res;

  float t = expf(z.x);

  sincosf(z.y, &res.y, &res.x);

  res.x *= t;

  res.y *= t;

  return res;
}


#endif
