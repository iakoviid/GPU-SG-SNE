#include "relocateData.cuh"
#include "utils_cuda.cuh"
#include <sys/time.h>
#include "cuda_runtime.h"

#include "sort.h"
#include <cmath>
//#include "sort.cuh"
__global__ void generateBoxIdx(uint64_t *Code, const coord *Y, coord scale,
                               const int nPts, const int nDim, const int nGrid,
                               const coord multQuant, const uint32_t qLevel) {
  register uint64_t C[3];
  register coord Yscale;
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
    return (uint32_t)C[TID];

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
// Concern about point 0
__global__ void gridSizeAndIdxKernel(uint32_t *ib, uint32_t *cb, uint64_t *C,
                                     int nPts, int nDim, int nGrid,
                                     uint32_t qLevel) {
  uint32_t idxCur;
  uint32_t idxNew;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    if (TID < nPts - 1) {
      idxNew = untangleLastDimDevice(nDim, TID, qLevel, C);
      idxCur = untangleLastDimDevice(nDim, TID + 1, qLevel, C);
      if (idxNew != idxCur) {
        ib[idxCur] = TID + 1;
      }
      if (idxCur - idxNew > 1) {
        ib[idxNew + 1] = TID + 1;
      }
    } else {
      idxNew = untangleLastDimDevice(nDim, TID, qLevel, C);
      if (idxNew != idxCur)
        ib[idxNew + 1] = TID + 1;
    }
  }
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < nGrid - 1; TID += gridDim.x * blockDim.x) {
    idxCur = ib[TID];
    idxNew = ib[TID + 1];
    cb[TID] = idxNew - idxCur;
  }
}

__global__ void gridIdxKernel(uint32_t *ib, uint32_t *cb, uint64_t *C, int nPts,
                              int nDim, int nGrid, uint32_t qLevel) {
  uint32_t idxCur;
  uint32_t idxNew;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {

    if (TID < nPts - 1) {
      idxNew = untangleLastDimDevice(nDim, TID, qLevel, C);
      idxCur = untangleLastDimDevice(nDim, TID + 1, qLevel, C);
      if (idxNew != idxCur) {
        ib[idxCur] = TID + 1;
      }
      if (idxCur - idxNew > 1) {
        ib[idxNew + 1] = TID + 1;
      }
    } else {
      idxNew = untangleLastDimDevice(nDim, TID, qLevel, C);
      if (idxNew != idxCur)
        ib[idxNew + 1] = TID + 1;
    }

    // atomicAdd(&cb[idxNew], 1);
  }
}
__global__ void gridSizeKernel(uint32_t *ib, uint32_t *cb,int nGrid){
  uint32_t idxCur;
  uint32_t idxNew;
    for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
         TID < nGrid - 1; TID += gridDim.x * blockDim.x) {
      idxCur = ib[TID];
      idxNew = ib[TID + 1];
      cb[TID] = idxNew - idxCur;
    }

}



void relocateCoarseGrid(
    coord *Yptr,                            // Scattered point coordinates
    uint32_t* iPerm, // Data relocation permutation
    uint32_t *ib, // Starting index of box (along last dimension)
    uint32_t *cb, // Number of scattered points per box (along last dimension)
    int nPts,     // Number of data points
    int nGrid,    // Grid dimensions (+1)
    int nDim      // Number of dimensions
) {
  coord *y_d = Yptr;
  thrust::device_ptr<coord> yVec_ptr(y_d);
  thrust::device_vector<coord> yVec_d(yVec_ptr, yVec_ptr + nPts * nDim);
  thrust::device_vector<coord>::iterator iter =
      thrust::max_element(yVec_d.begin(), yVec_d.end());
  unsigned int position = iter - yVec_d.begin();
  coord maxy = yVec_d[position];
  uint64_t *Codes;
  CUDA_CALL(cudaMallocManaged(&Codes, nPts * sizeof(uint64_t)));
  uint64_t *Codes2;
  CUDA_CALL(cudaMallocManaged(&Codes2, nPts * sizeof(uint64_t)));
  coord *yd2;
  CUDA_CALL(cudaMallocManaged(&yd2, nPts * nDim * sizeof(coord)));
  uint32_t *iPerm2;
  CUDA_CALL(cudaMallocManaged(&iPerm2, nPts * sizeof(uint32_t)));

  thrust ::device_ptr<uint64_t> Codes_ptr(Codes);
  coord multQuant = nGrid - 1 - std::numeric_limits<coord>::epsilon();

  uint32_t qLevel = 0;
  qLevel = ceil(log(nGrid) / log(2));
  generateBoxIdx<<<32, 256>>>(Codes, y_d, maxy, nPts, nDim, nGrid, multQuant,
                              qLevel);
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  radix_sort(Codes2, Codes, nPts,qLevel,(nDim - 1) * qLevel,iPerm2,iPerm,yd2,y_d,nDim);

  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("THRUST SORT time milliseconds %f\n", elapsedTime);



  gridIdxKernel<<<32, 256>>>(ib, cb, Codes2, nPts, nDim, nGrid, qLevel);
  cudaDeviceSynchronize();
  gridSizeKernel<<<32, 256>>>(ib, cb,nGrid );
  CUDA_CALL(cudaFree(Codes));
  CUDA_CALL(cudaFree(Codes2));
  //CUDA_CALL(cudaFree(y_d));
  iPerm=iPerm2;
  y_d = yd2;

  return;
}
