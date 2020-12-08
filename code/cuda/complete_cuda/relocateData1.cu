#include "relocateData.cuh"
#include <sys/time.h>
#define LIMIT_E 0.00000000000001
__global__ void generateBoxIdx(uint32_t *Code, const coord *Y, coord scale,
                               const int nPts, const int nDim, const int nGrid,
                               const coord multQuant, const uint32_t qLevel) {
  register uint32_t C[3];
  register coord Yscale;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (int j = 0; j < nDim; j++) {
      Yscale = Y[TID + j * nPts] / scale;
      if (Yscale >= 1) {
        Yscale = 1 - LIMIT_E;
        // printf("Yscale= %lf\n",Yscale );
      }
      C[j] = (uint64_t)abs(floor(multQuant * Yscale));
    }
    switch (nDim) {

    case 1:
      Code[TID] = C[0];

    case 2:
      Code[TID] = C[1] * nGrid + C[0];

    case 3:
      Code[TID] = C[2] * nGrid * nGrid + C[1] * nGrid + C[0];
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

    // atomicAdd(&cb[idxNew], 1);
  }

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < nGrid - 1; TID += gridDim.x * blockDim.x) {
    idxCur = ib[TID];
    idxNew = ib[TID + 1];

    cb[TID] = (uint32_t)(idxNew - idxCur);
  }
}

__global__ void gridSizeAndIdxKernelNew(uint32_t *ib, uint32_t *cb, uint64_t *C,
                                        int nPts, int nDim, int nGrid,
                                        uint32_t qLevel) {
  uint32_t idxCur = -1;
  uint32_t idxNew;
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    idxNew = untangleLastDimDevice(nDim, TID, qLevel, C);
    if (TID > 0) {
      idxCur = untangleLastDimDevice(nDim, TID - 1, qLevel, C);
    }
    if (idxNew != idxCur)
      ib[idxNew + 1] = TID + 1;
    // atomicAdd(&cb[idxNew], 1);
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
__global__ void gridSizeKernel(uint32_t *ib, uint32_t *cb, int nGrid) {
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
    thrust::device_vector<uint32_t> &iPerm, // Data relocation permutation
    uint32_t *ib, // Starting index of box (along last dimension)
    uint32_t *cb, // Number of scattered points per box (along last dimension)
    int nPts,     // Number of data points
    int nGrid,    // Grid dimensions (+1)
    int nDim      // Number of dimensions
) {
  uint32_t Blocks=64;
  uint32_t threads=512;
  coord *y_d = Yptr;
  thrust::device_ptr<coord> yVec_ptr(y_d);
  thrust::device_vector<coord> yVec_d(yVec_ptr, yVec_ptr + nPts * nDim);
  thrust::device_vector<coord>::iterator iter =
      thrust::max_element(yVec_d.begin(), yVec_d.end());
  unsigned int position = iter - yVec_d.begin();
  coord maxy = yVec_d[position];
  uint32_t *Codes;
  CUDA_CALL(cudaMallocManaged(&Codes, nPts * sizeof(uint32_t)));
  thrust ::device_ptr<uint32_t> Codes_ptr(Codes);
  coord multQuant = nGrid - 1 - std::numeric_limits<coord>::epsilon();

  uint32_t qLevel = 0;
  qLevel = ceil(log(nGrid) / log(2));
  generateBoxIdx<<<Blocks, threads>>>(Codes, y_d, maxy, nPts, nDim, nGrid, multQuant,
                              qLevel);
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);

  switch (nDim) {

  case 1:
    thrust ::stable_sort_by_key(
        Codes_ptr, Codes_ptr + nPts,
        make_zip_iterator(make_tuple(yVec_ptr, iPerm.begin())));

  case 2:
    thrust ::stable_sort_by_key(Codes_ptr, Codes_ptr + nPts,
                                make_zip_iterator(make_tuple(
                                    yVec_ptr, yVec_ptr + nPts, iPerm.begin())));

  case 3:
    thrust ::stable_sort_by_key(
        Codes_ptr, Codes_ptr + nPts,
        make_zip_iterator(make_tuple(yVec_ptr, yVec_ptr + nPts,
                                     yVec_ptr + 2 * nPts, iPerm.begin())));
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("THRUST SORT time milliseconds %f\n", elapsedTime);
/*
  gridIdxKernel<<<Blocks, threads>>>(ib, cb, Codes, nPts, nDim, nGrid, qLevel);
  cudaDeviceSynchronize();
  gridSizeKernel<<<Blocks, threads>>>(ib, cb, nGrid);
*/
   CUDA_CALL(cudaFree(Codes));

  return;
}
