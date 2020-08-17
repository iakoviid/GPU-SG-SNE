#include "relocateData.cuh"
#include <sys/time.h>
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

__global__ void Block_binning(uint64_t *Cs, uint32_t m, uint32_t n, uint32_t sft,
                              uint32_t *BinCursor, uint64_t mask,
                              uint32_t nBin) {
  uint32_t block = blockIdx.x;
  int size = ((block + 1) * m < n) ? m : (n - block * m);

  for (uint32_t TID = threadIdx.x; TID < size; TID += blockDim.x) {
    uint32_t const ii = (Cs[block * m + TID] >> sft) & mask;
    atomicAdd(&BinCursor[block * nBin + ii], 1);
  }
}
template <typename dataval>
__global__ void translation_level(dataval *Ys, dataval *Yt, uint64_t *Ct,
                                  uint64_t *Cs, uint32_t *Pt, uint32_t *Ps,
                                  int m, uint32_t n, uint32_t d, uint32_t sft,
                                  uint32_t *BinCursor, uint64_t mask,
                                  uint32_t nBin) {
  uint32_t block = blockIdx.x;
  int size = ((block + 1) * m < n) ? m : (n - block * m);

  for (uint32_t TID = threadIdx.x; TID < size; TID += blockDim.x) {
    uint32_t const idx = block * m + TID;
    uint32_t const ii = (Cs[idx] >> sft) & mask;

    uint32_t const jj = BinCursor[block * nBin + ii];
    Ct[jj] = Cs[idx];
    for (int k = 0; k < d; k++) {
      Yt[jj + k * n] = Ys[idx + k * n];
    }
    Pt[jj] = Ps[idx];
    BinCursor[block * nBin + ii]++;
  }
}

template <typename dataval>
void multilevel_translation(uint64_t *const Cs, uint64_t *const Ct,
                            uint32_t *const Ps, uint32_t *const Pt,
                            dataval *const Ys, dataval *const Yt,
                            uint32_t prev_off, const uint32_t nbits,
                            uint32_t sft, const uint32_t n, const uint32_t d,
                            uint32_t nb,
                            uint32_t nBlocks) { // prepare bins
  uint32_t nBin = (0x01 << (nbits));
  int m = (int)std::ceil((float)n / (float)nBlocks);

  uint32_t *BinCursor;
  CUDA_CALL(cudaMallocManaged(&BinCursor, nBin * nBlocks * sizeof(uint32_t)));
  cudaMemset(BinCursor, 0, nBin * nBlocks*sizeof(uint32_t));

  // get mask for required number of bits
  uint64_t mask = (0x01 << (nbits)) - 1;

  Block_binning<<<32, 256>>>(Cs, m, n, sft, BinCursor, mask, nBin);
  cudaDeviceSynchronize();
  int offset = 0;
  for (int i=0; i<nBin; i++){
    for(int j=0; j<nBlocks; j++) {
      int const ss = BinCursor[j*nBin + i];
      BinCursor[j*nBin + i] = offset;
      offset += ss;
    }
  }
  translation_level<<<32, 1>>>(Ys, Yt, Ct, Cs, Pt, Ps, m, n, d, sft, BinCursor,
                               mask, nBin);

  uint32_t *BinCursor2;
  CUDA_CALL(cudaMallocManaged(&BinCursor2, nBin * nBlocks * sizeof(uint32_t)));
  uint32_t offset2=0;
  uint32_t nPts;
  while (sft >= nbits) {

    for (int i = 0; i < nBin; i++) {
      cudaMemset(BinCursor2, 0, nBin * nBlocks*sizeof(uint32_t));

      nPts=BinCursor[(nBlocks-1)*nBin+i]-offset2;

      m = (int)std::ceil((float)nPts / (float)nBlocks);

      Block_binning<<<32, 256>>>(&Cs[offset], m, nPts, sft, BinCursor2, mask, nBin);
      cudaDeviceSynchronize();
       offset = 0;
      for (int k=0; i<nBin; i++){
        for(int j=0; j<nBlocks; j++) {
          int const ss = BinCursor[j*nBin + k];
          BinCursor[j*nBin + k] = offset;
          offset += ss;
        }
      }
      translation_level<<<32, 1>>>(&Ys[offset2], &Yt[offset2], &Ct[offset2], &Cs[offset2], &Pt[offset2],&Ps[offset2], m, nPts, d, sft,
                                   BinCursor2, mask, nBin);

      offset2=BinCursor[(nBlocks-1)*nBin+i];
    }
    sft -= nbits;
  }
}


__global__ void permute(uint32_t* Cs,uint32_t n,uint32_t m,uint32_t sft, uint64_t mask,uint32_t nBin)
{
  extern __shared__ uint32_t Arrays[];
  uint32_t* tBc=Arrays;
  uint32_t* tBIdx=&Arrays[nBin*blockDim.x];

  uint32_t block = blockIdx.x;
  int size = ((block + 1) * m < n) ? m : (n - block * m);

  for (uint32_t TID = threadIdx.x; TID < size; TID += blockDim.x) {
    uint32_t const idx = block * m + TID;
    uint32_t const ii = (Cs[idx] >> sft) & mask;
    tBc[TID*nBin+ii]++;
    tBIdx[tBc[ii]]=idx;

  }



}
template <typename dataval>
void MultilevelTranslation(uint64_t *const Cs, uint64_t *const Ct,
                            uint32_t *const Ps, uint32_t *const Pt,
                            dataval *const Ys, dataval *const Yt,
                            uint32_t prev_off, const uint32_t nbits,
                            uint32_t sft, const uint32_t n, const uint32_t d,
                            uint32_t nb,
                            uint32_t nBlocks){

  uint32_t nBin = (0x01 << (nbits));
  uint32_t m = (uint32_t)std::ceil((float)n / (float)nBlocks);

  uint32_t *BinCursor;
  CUDA_CALL(cudaMallocManaged(&BinCursor, nBin * nBlocks * sizeof(uint32_t)));
  cudaMemset(BinCursor, 0, nBin * nBlocks*sizeof(uint32_t));

  // get mask for required number of bits
  uint64_t mask = (0x01 << (nbits)) - 1;

  Block_binning<<<nBlocks, 256>>>(Cs, m, n, sft, BinCursor, mask, nBin);
  cudaDeviceSynchronize();
  thrust::exclusive_scan(thrust::device, BinCursor, BinCursor + nBin * nBlocks,
                         BinCursor, 0);
  uint32_t threads=64;
  uint32_t byteSize= threads*nBin*(ceil(m/threads)+1)*sizeof(uint32_t);
  //permute<<<nBlocks,threads,512*sizeof(uint32_t)>>();



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
  multilevel_translation(Codes, Codes2, thrust::raw_pointer_cast(iPerm.data()),
                         iPerm2, y_d, yd2, (uint32_t)0, qLevel,
                         (uint32_t)qLevel * (nDim - 1), (uint32_t)nPts,
                         (uint32_t)nDim, (uint32_t)nGrid, 32);

  /*
    switch (nDim) {

    case 1:
      thrust ::stable_sort_by_key(
          Codes_ptr, Codes_ptr + nPts,
          make_zip_iterator(make_tuple(yVec_ptr, iPerm.begin())));

    case 2:
      thrust ::stable_sort_by_key(Codes_ptr, Codes_ptr + nPts,
                                  make_zip_iterator(make_tuple(
                                      yVec_ptr, yVec_ptr + nPts,
    iPerm.begin())));

    case 3:
      thrust ::stable_sort_by_key(
          Codes_ptr, Codes_ptr + nPts,
          make_zip_iterator(make_tuple(yVec_ptr, yVec_ptr + nPts,
                                       yVec_ptr + 2 * nPts, iPerm.begin())));
    }*/
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("THRUST SORT time milliseconds %f\n", elapsedTime);

  gridSizeAndIdxKernel<<<32, 256>>>(ib, cb, Codes, nPts, nDim, nGrid, qLevel);
  CUDA_CALL(cudaFree(Codes));
  y_d=yd2;

  return;
}
