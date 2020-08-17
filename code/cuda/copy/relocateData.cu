#include "relocateData.cuh"
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

void relocateCoarseGrid(
    coord *Yptr,                            // Scattered point coordinates
    thrust::device_vector<uint32_t> &iPerm, // Data relocation permutation
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
  thrust ::device_ptr<uint64_t> Codes_ptr(Codes);
  coord multQuant = nGrid - 1 - std::numeric_limits<coord>::epsilon();

  uint32_t qLevel = 0;
  qLevel = ceil(log(nGrid) / log(2));
  generateBoxIdx<<<32, 256>>>(Codes, y_d, maxy, nPts, nDim, nGrid, multQuant,
                              qLevel);

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

  gridSizeAndIdxKernel<<<32, 256>>>(ib, cb, Codes, nPts, nDim, nGrid, qLevel);

  return;
}
