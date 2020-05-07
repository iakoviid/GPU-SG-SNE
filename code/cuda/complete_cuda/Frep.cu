
_global__ void compute_repulsive_forces_kernel(
    volatile coord *__restrict__ frep, const coord *const Y,
    const int num_points, const int nDim, const coord *const Phi,
    volatile coord *__restrict__ zetaVec, uint32_t *iPerm) {
  register coord Ysq = 0;
  register coord z = 0;

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x;
       TID < num_points; TID += gridDim.x * blockDim.x) {

    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[TID + j * num_points] * Y[TID + j * num_points];
      z -= 2 * Y[TID + j * num_points] * Phi[TID * (nDim + 1) + j + 1];
    }
    z += (1 + 2 * Ysq) * Phi[TID * (nDim + 1)];
    zetaVec[TID] = z;
    for (uint32_t j = 0; j < nDim; j++) {
      frep[iPerm[TID] + j * num_points] =
          Y[TID + j * num_points] * Phi[TID * (nDim + 1)] -
          Phi[TID * (nDim + 1) + j + 1];
    }
  }
}
coord zetaAndForce(coord *Ft_d, coord *y_d, int n, int d, coord *Phi,
                    thrust::device_vector<uint32_t> &iPerm,
                    thrust::device_vector<coord> &zetaVec) {

  int threads = 1024;
  int Blocks = 64;
  compute_repulsive_forces_kernel<<<Blocks, threads>>>(
      Ft_d, y_d, n, d, Phi, thrust::raw_pointer_cast(zetaVec.data()),
      thrust::raw_pointer_cast(iPerm.data()));
  coord z = thrust::reduce(zetaVec.begin(), zetaVec.end()) - n;
  return z;
}
coord computeFrepulsive_interp(coord *Frep, coord *y, int n, int d, double h,
                               int np) {

  // ~~~~~~~~~~ make temporary data copies
  coord *yt = static_cast<coord *>(malloc(n * d * sizeof(coord)));
  coord *yr = static_cast<coord *>(malloc(n * d * sizeof(coord)));

  // struct timeval start;

  // ~~~~~~~~~~ move data to (0,0,...)
  coord miny[d];
  for (int i = 0; i < d; i++) {
    miny[i] = std::numeric_limits<coord>::infinity();
  }

  //--G cauch is translationaly invariant
  for (int i = 0; i < n; i++)
    for (int j = 0; j < d; j++)
      miny[j] = miny[j] > y[i * d + j] ? y[i * d + j] : miny[j];

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      y[i * d + j] -= miny[j];
    }
  }

  // ~~~~~~~~~~ find maximum value (across all dimensions) and get grid size
  //--G I have something similar max(maxy/h,14) vs max((maxy-miny)*2,20)
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;

  int nGrid = std::max((int)std::ceil(maxy / h), 14);
  nGrid = getBestGridSize(nGrid);

  //#ifdef VERBOSE
  std::cout << "Grid: " << nGrid << " h: " << h << "maxy: " << maxy
            << std::endl;
  //#endif

  // ~~~~~~~~~~ setup inputs to nuConv

  std::copy(y, y + (n * d), yt);

  coord *VScat = (coord *)malloc(n * (d + 1) * sizeof(coord));
  coord *PhiScat = (coord *)calloc(n * (d + 1), sizeof(coord));
  uint32_t *iPerm = (uint32_t *)malloc(n * sizeof(uint32_t));
  uint32_t *ib = (uint32_t *)calloc(nGrid, sizeof(uint32_t));
  uint32_t *cb = (uint32_t *)calloc(nGrid, sizeof(uint32_t));

  for (int i = 0; i < n; i++) {
    iPerm[i] = i;
  }

  // start = tsne_start_timer();
  // relocateCoarseGrid(&yt, &iPerm, ib, cb, n, nGrid, d, np);
  /*
  if (timeInfo != nullptr)
    timeInfo[0] = tsne_stop_timer("Gridding", start);
  else
    tsne_stop_timer("Gridding", start);
    */
  // ----- setup VScat (value on scattered points)

  for (int i = 0; i < n; i++) {

    VScat[i * (d + 1)] = 1.0;
    for (int j = 0; j < d; j++)
      VScat[i * (d + 1) + j + 1] = yt[i * d + j];
  }

  std::copy(yt, yt + (n * d), yr);

  // ~~~~~~~~~~ run nuConv
  /*
  if (timeInfo != nullptr)
    nuconv(PhiScat, yt, VScat, ib, cb, n, d, d + 1, np, nGrid, &timeInfo[1]);
  else
    nuconv(PhiScat, yt, VScat, ib, cb, n, d, d + 1, np, nGrid);
  */
  // ~~~~~~~~~~ compute Z and repulsive forces

  // start = tsne_start_timer();
  coord zeta = zetaAndForce(Frep, yr, PhiScat, iPerm, n, d);
  /*
  if (timeInfo != NULL)
    timeInfo[4] = tsne_stop_timer("F&Z", start);
  else
    tsne_stop_timer("F&Z", start);
  */
  free(yt);
  free(yr);
  free(VScat);
  free(PhiScat);
  free(iPerm);
  free(ib);
  free(cb);
  return zeta;
}
