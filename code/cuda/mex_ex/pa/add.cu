#include "cuda_runtime.h"
#include "add_wrapper.hpp"

__global__ void addKernel(float *c, const float *a, const float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// C = A + B
void addWithCUDA(float *cpuC, const float *cpuA, const float *cpuB, const size_t sz)
{    
    //TODO: add error checking
    
    // choose which GPU to run on
    cudaSetDevice(0);
    
    // allocate GPU buffers
    float *gpuA, *gpuB, *gpuC;
    cudaMalloc((void**)&gpuA, sz*sizeof(float));
    cudaMalloc((void**)&gpuB, sz*sizeof(float));
    cudaMalloc((void**)&gpuC, sz*sizeof(float));
    
    // copy input vectors from host memory to GPU buffers
    cudaMemcpy(gpuA, cpuA, sz*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(gpuB, cpuB, sz*sizeof(float), cudaMemcpyHostToDevice);
    
    // launch kernel on the GPU with one thread per element
    addKernel<<<1,sz>>>(gpuC, gpuA, gpuB);
    
    // wait for the kernel to finish
    cudaDeviceSynchronize();
    
    // copy output vector from GPU buffer to host memory
    cudaMemcpy(cpuC, gpuC, sz*sizeof(float), cudaMemcpyDeviceToHost);
    
    // cleanup
    cudaFree(gpuA);
    cudaFree(gpuB);
    cudaFree(gpuC);
}

void resetDevice()
{
    cudaDeviceReset();
}
