/*Heterogeneous computing is about efficiently using all processors in the system, 
including CPUs and GPUs. To do this, applications must execute functions concurrently
on multiple processors. CUDA Applications manage concurrency by executing asynchronous 
commands in streams, sequences of commands that execute in order. Different streams may
 execute their commands concurrently or out of order with respect to each other. 
[See the post How to Overlap Data Transfers in CUDA C/C++ for an example
When you execute asynchronous CUDA commands without specifying a stream, 
the runtime uses the default stream.

In this post I’ll show you how this can simplify achieving concurrency between kernels and data copies in CUDA programs.

As described by the CUDA C Programming Guide, asynchronous commands return control to the calling host thread
before the device has finished the requested task (they are non-blocking).

Let’s look at a trivial example. The following code simply launches eight copies of a simple kernel on eight streams.
 We launch only a single thread block for each grid so there are plenty of resources to run multiple of them concurrently.
As an example of how the legacy default stream causes serialization, 
we add dummy kernel launches on the default stream that do no work.
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
const int N = 1 << 20;

__global__ void kernel(float *x, int n)
{
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    for (int i = tid; i < n; i += blockDim.x * gridDim.x) {
        x[i] = sqrt(pow(3.14159,i));
    }
}

//two commands from different streams cannot run concurrently if the host thread issues any CUDA command to the default stream between them.
// the per-thread default stream, that has two effects. First, it gives each host thread its own default stream.
//This means that commands issued to the default stream by different host threads can run concurrently.
//Second, these default streams are regular streams.
//This means that commands in the default stream may run concurrently with commands in non-default streams.
int main()
{
    const int num_streams = 8;

    cudaStream_t streams[num_streams];
    float *data[num_streams];

    for (int i = 0; i < num_streams; i++) {
        cudaStreamCreate(&streams[i]);
 
        cudaMalloc(&data[i], N * sizeof(float));
        
        // launch one worker kernel per stream
        kernel<<<1, 64, 0, streams[i]>>>(data[i], N);

        // launch a dummy kernel on the default stream
        kernel<<<1, 1>>>(0, 0);
    }

    cudaDeviceReset();

    return 0;
}
