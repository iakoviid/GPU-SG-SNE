
#include "nuconv.cuh"
#define E_LIMIT 0.00000000000001
__global__ void Scale(coord *y, uint32_t nPts, uint32_t ng, uint32_t d,
                          coord maxy) {
  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts;
       TID += gridDim.x * blockDim.x) {
    for (int dim = 0; dim < d; dim++) {
      y[TID + dim * nPts] /= maxy;
      if (y[TID + dim * nPts] == 1) {
        y[TID + dim * nPts] = y[TID + dim * nPts] - E_LIMIT;
      }
      y[TID + dim * nPts] *= (ng - 3);
    }
  }
}

void nuconv(coord *PhiScat, coord *y, coord *VScat, uint32_t *ib, uint32_t *cb,
            int n, int d, int m, int nGridDim) {

  // ~~~~~~~~~~ Scale coordinates (inside bins)
  thrust::device_ptr<double> yVec_ptr(y);
  thrust::device_vector<double> yVec_d(yVec_ptr, yVec_ptr + n * d);
  thrust::device_vector<double>::iterator iter =
      thrust::max_element(yVec_d.begin(), yVec_d.end());
  unsigned int position = iter - yVec_d.begin();
  coord maxy = yVec_d[position];

  coord h = maxy / (nGridDim - 1 - std::numeric_limits<coord>::epsilon());

  // ~~~~~~~~~~ scat2grid
  int szV = pow(nGridDim + 2, d) * m;
  coord *VGrid;
  CUDA_CALL(cudaMallocManaged(&VGrid, szV * sizeof(coord)));
  Scale<<<64,512>>>(y, n, nGridDim, d, maxy);
  cudaDeviceSynchronize();
  switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD) {

      s2g1d<<<32, 512>>>(VGrid, y, VScat, nGridDim + 2, n, d, d + 1);
    } else {
      s2g1drb<<<64, 32>>>(VGrid, y, VScat, ib, cb, nGridDim + 2, n, d, m);
    }

    // s2g1drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD)

      s2g2d<<<32, 512>>>(VGrid, y, VScat, nGridDim + 2, n, d, m);

    else

      s2g2drb<<<64,32>>>(VGrid, y, VScat, ib, cb, nGridDim + 2, n, d, m);
    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
       s2g3d<<<64,256>>>( VGrid, y, VScat, nGridDim+2, n, d, m );
    else
        s2g3drb<<<64,32>>>(VGrid, y, VScat, ib, cb, nGridDim + 2, n, d, m);

    // s2g3drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;
  }

  // ~~~~~~~~~~ grid2grid
  coord *PhiGrid;
  CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(coord)));

  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = nGridDim + 2;
  }

  switch (d) {

  case 1:
    conv1dnopadcuda(PhiGrid, VGrid, h, nGridDim + 2, m, d);

    break;

  case 2:
    conv2dnopadcuda(PhiGrid, VGrid, h, nGridDims, m, d);

    break;

  case 3:
    conv3dnopadcuda(PhiGrid,VGrid,h,nGridDims,m,d);
    break;
  }

  // ~~~~~~~~~~ grid2scat

  switch (d) {

  case 1:
    g2s1d<<<32, 256>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);

    break;

  case 2:
    g2s2d<<<32, 256>>>(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    break;

  case 3:
    g2s3d<<<32, 256>>>( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;
  }

  // ~~~~~~~~~~ deallocate memory
}
