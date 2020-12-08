
__global__ void Block_binning(uint64_t *Cs, uint32_t n, uint32_t sft,
                              uint32_t *BinCursor, uint64_t mask, uint32_t nBin,
                              uint32_t *keys) {

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    uint32_t const ii = (Cs[TID] >> sft) & mask;
    keys[TID] = ii;
    atomicAdd(&BinCursor[ii], 1);
  }
}
template <typename dataval>
__global__ void
permuteData(uint64_t *const Cs, uint64_t *const Ct, uint32_t *const Ps,
            uint32_t *const Pt, dataval *const Ys, dataval *const Yt,
            const uint32_t n, const uint32_t d, uint32_t *BinCursor,
            uint32_t *idx, uint32_t sft, uint64_t mask) {
  for (uint32_t TID = threadIdx.x; TID < n; TID += blockDim.x) {
    uint32_t const ii = (Cs[TID] >> sft) & mask;
    uint32_t const jj = BinCursor[ii] + idx[TID];

    Ct[jj] = Cs[TID];
    for (int k = 0; k < d; k++) {
      Yt[jj + k * n] = Ys[TID + k * n];
    }
    Pt[jj] = Ps[TID];
  }
}



template <typename dataval>
void multilevel_translation(uint64_t * Cs, uint64_t * Ct,
                            uint32_t * Ps, uint32_t * Pt,
                            dataval * Ys, dataval * Yt,
                            uint32_t prev_off, const uint32_t nbits,
                            uint32_t sft, const uint32_t n, const uint32_t d,
                            uint32_t nb) { // prepare bins
  // prepare bins

  uint32_t nBin = (0x01 << (nbits));
  uint32_t *BinCursor;
  CUDA_CALL(cudaMallocManaged(&BinCursor, nBin * sizeof(uint32_t)));
  uint32_t *key;
  CUDA_CALL(cudaMallocManaged(&key, n * sizeof(uint32_t)));
  uint32_t *idx;
  CUDA_CALL(cudaMallocManaged(&idx, n * sizeof(uint32_t)));
  uint64_t mask = (0x01 << (nbits)) - 1;
  uint32_t* temp;
  coord* coord_temp;
  uint64_t* temp64;

  //while(sft>=nbits){

  initKernel<<<64, 256>>>(idx, (uint32_t)1, n);
  cudaMemset(BinCursor, 0,  nBin * sizeof(uint32_t));
  // Create Histogram
  Block_binning<<<32, 256>>>(Cs, n, sft, BinCursor, mask, nBin, key);
  cudaDeviceSynchronize();
  thrust::exclusive_scan(thrust::device, BinCursor, BinCursor + n, BinCursor,
                         0);
  thrust::exclusive_scan_by_key(key, key + n, idx, idx); // in-place scan
  permuteData<<<64, 256>>>(Cs, Ct, Ps, Pt, Ys, Yt, n, d, BinCursor, idx, sft,
                           mask);
  sft-=nbits;
  if(0>1){
  temp64=Cs;
  Cs=Ct;
  Ct=temp64;

  temp=Ps;
  Ps=Pt;
  Pt=temp;

  coord_temp=Ys;
  Ys=Yt;
  Yt=coord_temp;}



                //}
}
