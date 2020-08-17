#include <cuda_runtime.h>
#include <stdio.h>


__global__ void k_testLocking(unsigned int* locks, int n) {
    int id = threadIdx.x /32;
    bool leaveLoop = false;
    while (!leaveLoop) {
        if (atomicExch(&(locks[id]), 1u) == 0u) {
            printf("threadID %d\n",threadIdx.x );
            leaveLoop = true;
            atomicExch(&(locks[id]),0u);
        }
    }
}

int main(int argc, char** argv) {
    //initialize the locks array on the GPU to (0...0)
    unsigned int* locks;
    unsigned int zeros[10]; for (int i = 0; i < 10; i++) {zeros[i] = 0u;}
    cudaMalloc((void**)&locks, sizeof(unsigned int)*10);
    cudaMemcpy(locks, zeros, sizeof(unsigned int)*10, cudaMemcpyHostToDevice);

    //Run the kernel:
    k_testLocking<<<dim3(1), dim3(256)>>>(locks, 10);

    //Check the error messages:
    cudaError_t error = cudaGetLastError();
    cudaFree(locks);
    if (cudaSuccess != error) {
        printf("error 1: CUDA ERROR (%d) {%s}\n", error, cudaGetErrorString(error));
        exit(-1);
    }
}
