#include "cuda_runtime.h"
#include "relocateData.cuh"
#include "utils_cuda.cuh"
#include <sys/time.h>

#include "sort.h"
#include <cmath>
extern cudaStream_t streamRep;

//#include "sort.cuh"
template <class dataPoint>
__global__ void generateBoxIdx(uint64_t *Code, const dataPoint *Y, dataPoint scale,
                               const int nPts, const int nDim, const int nGrid,
                               const dataPoint multQuant, const uint32_t qLevel) {
  register uint64_t C[3];
  register dataPoint Yscale;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (int j = 0; j < nDim; j++) {
      Yscale = Y[TID + j * nPts] / scale;
      if (Yscale >= 1) {
        Yscale = 1 - 0.00000000000001;
        // printf("Yscale= %lf\n",Yscale );
      }
      C[j] = (uint64_t)abs(floor(multQuant * Yscale));
    }
    switch (nDim) {

    case 1:
      Code[TID] = (uint64_t)C[0];

    case 2:
      Code[TID] = (((uint64_t)C[1]) << qLevel) | (((uint64_t)C[0]));

    case 3:
      Code[TID] = (((uint64_t)C[2]) << 2 * qLevel) |
                  (((uint64_t)C[1]) << qLevel) | ((uint64_t)C[0]);
    }
  }
  return;
}

__inline__ __device__ uint32_t untangleLastDimDevice(int nDim, int TID,
                                                     uint32_t qLevel,
                                                     uint64_t *C) {
  uint64_t mask;
  switch (nDim) {
  case 1:
    mask = (1 << qLevel) - 1;
    return (uint32_t)(C[TID] & mask);

  case 2: {
    mask = (1 << 2 * qLevel) - 1;

    return (uint32_t)((C[TID] & mask) >> qLevel);
  }

  case 3: {
    mask = (1 << 3 * qLevel) - 1;

    return (uint32_t)((C[TID] & mask) >> 2 * qLevel);
  }
  }
  return 0;
}

__inline__ __device__ uint32_t gridIdxKernel(int nDim, int TID, uint32_t qLevel,
                                             uint64_t *C, int nGrid) {
  switch (nDim) {
  case 1:
    return untangleLastDimDevice(1, TID, qLevel, C);
  case 2:
    return untangleLastDimDevice(2, TID, qLevel, C) * nGrid +
           untangleLastDimDevice(1, TID, qLevel, C);
  case 3:
    return untangleLastDimDevice(3, TID, qLevel, C) * nGrid * nGrid +
           untangleLastDimDevice(2, TID, qLevel, C) * nGrid +
           untangleLastDimDevice(1, TID, qLevel, C);
  }
}

__global__ void gridIdxKernel(uint32_t *ib, uint32_t *cb, uint64_t *C, int nPts,
                              int nDim, int nGrid, uint32_t qLevel) {
  uint32_t idxCur;
  uint32_t idxNew;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    if (TID < nPts - 1) {
      idxNew = gridIdxKernel(nDim, TID, qLevel, C, nGrid - 1);
      idxCur = gridIdxKernel(nDim, TID + 1, qLevel, C, nGrid - 1);
      if (idxNew != idxCur) {
        ib[idxCur] = TID + 1;
      }
      if (idxCur - idxNew > 1) {
        ib[idxNew + 1] = TID + 1;
      }
    } else {
      idxNew = gridIdxKernel(nDim, TID, qLevel, C, nGrid - 1);
      if (idxNew != idxCur)
        ib[idxNew + 1] = TID + 1;
    }

    // atomicAdd(&cb[idxNew], 1);
  }
}
__global__ void gridSizeKernel(uint32_t *ib, uint32_t *cb, int nGrid) {
  uint32_t idxCur;
  uint32_t idxNew;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < nGrid * nGrid - 1; TID += gridDim.x * blockDim.x) {
    idxCur = ib[TID];
    idxNew = ib[TID + 1];
    cb[TID] = idxNew - idxCur;
  }
}

__global__ void gridIdxKernelnew(int *ib, uint64_t *C, int nPts, int nDim,
                                 int nGrid, uint32_t qLevel) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x + 1; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    uint32_t idxCur = gridIdxKernel(nDim, TID, qLevel, C, nGrid - 1);
    uint32_t idxPrev = gridIdxKernel(nDim, TID - 1, qLevel, C, nGrid - 1);
    if (idxCur != idxPrev) {
      ib[idxCur] = TID;
      if (idxPrev + 1 != idxCur) {
        ib[idxPrev + 1] = TID;
      }
    }
    if (TID == nPts - 1) {
      ib[idxCur + 1] = nPts;
    }
    if (TID == 1) {
      ib[gridIdxKernel(nDim, 0, qLevel, C, nGrid - 1)] = 0;
    }
  }
}
template <class dataPoint>
void relocateCoarseGrid(dataPoint *Yptr,     // Scattered point dataPointinates
                        uint32_t *iPerm, // Data relocation permutation
                        int *ib, // Starting index of box (along last dimension)
                        int nPts,  // Number of data points
                        int nGrid, // Grid dimensions (+1)
                        int nDim   // Number of dimensions
) {
  dataPoint *y_d = Yptr;
  thrust::device_ptr<dataPoint> yVec_ptr(y_d);
  dataPoint maxy =
      thrust::reduce(thrust::cuda::par.on(streamRep), yVec_ptr,
                     yVec_ptr + nPts * nDim, 0.0, thrust::maximum<dataPoint>());
  uint64_t *Codes;
  CUDA_CALL(cudaMallocManaged(&Codes, nPts * sizeof(uint64_t)));
  uint64_t *Codes2;
  CUDA_CALL(cudaMallocManaged(&Codes2, nPts * sizeof(uint64_t)));
  dataPoint *yd2;
  CUDA_CALL(cudaMallocManaged(&yd2, nPts * nDim * sizeof(dataPoint)));
  uint32_t *iPerm2;
  CUDA_CALL(cudaMallocManaged(&iPerm2, nPts * sizeof(uint32_t)));

  thrust ::device_ptr<uint64_t> Codes_ptr(Codes);
  dataPoint multQuant = nGrid - 1 - std::numeric_limits<dataPoint>::epsilon();

  uint32_t qLevel = 0;
  qLevel = ceil(log(nGrid) / log(2));
  generateBoxIdx<<<32, 256>>>(Codes, y_d, maxy, nPts, nDim, nGrid, multQuant,
                              qLevel);
  radix_sort(Codes2, Codes, nPts, qLevel, (nDim - 1) * qLevel, iPerm2, iPerm,
             yd2, y_d, nDim);

  uint32_t points = pow(nGrid - 1, nDim) + 1;

  CUDA_CALL(cudaMemset(ib, -1, points * (sizeof(int))));
  gridIdxKernelnew<<<32, 256>>>(ib, Codes2, nPts, nDim, nGrid, qLevel);
  // gridSizeKernel<<<32, 256>>>(ib, cb,nGrid );
  CUDA_CALL(cudaFree(Codes));
  CUDA_CALL(cudaFree(Codes2));
  CUDA_CALL(cudaFree(iPerm2));
  CUDA_CALL(cudaFree(yd2));

  iPerm = iPerm2;
  y_d = yd2;

  return;
}
template
void relocateCoarseGrid(float *Yptr,     // Scattered point coordinates
                        uint32_t *iPerm, // Data relocation permutation
                        int *ib, // Starting index of box (along last dimension)
                        int nPts,  // Number of data points
                        int nGrid, // Grid dimensions (+1)
                        int nDim   // Number of dimensions
);
template
void relocateCoarseGrid(double *Yptr,     // Scattered point coordinates
                        uint32_t *iPerm, // Data relocation permutation
                        int *ib, // Starting index of box (along last dimension)
                        int nPts,  // Number of data points
                        int nGrid, // Grid dimensions (+1)
                        int nDim   // Number of dimensions
);
