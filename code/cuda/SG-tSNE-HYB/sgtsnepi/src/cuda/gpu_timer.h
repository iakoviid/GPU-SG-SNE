/*!
  \file   gpu_timer.h
  \brief  Simple GPU timer.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/

#ifndef GPU_TIMER_H__
#define GPU_TIMER_H__

#include <cuda_runtime.h>

struct GpuTimer
{
  cudaEvent_t start;
  cudaEvent_t stop;

  GpuTimer()
  {
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
  }

  ~GpuTimer()
  {
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
  }

  void Start(cudaStream_t stream)
  {
    cudaEventRecord(start, stream);
  }

  void Stop(cudaStream_t stream)
  {
    cudaEventRecord(stop, stream);
  }

  float Elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed, start, stop);
    return elapsed;
  }
};

#endif  /* GPU_TIMER_H__ */
