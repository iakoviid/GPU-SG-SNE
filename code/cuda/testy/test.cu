#include <cuda.h>
#include <cuda_runtime.h>
#include <limits>
#include <math.h>
#include <numeric>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>
#define LIMIT_SEQ 512
void conv1dnopad(double *const PhiGrid, const double *const VGrid,
                 const double h, uint32_t *const nGridDims, const uint32_t nVec,
                 const uint32_t nDim, const uint32_t nProc);

#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

double *generateRandomCoord(int n, int d) {

  double *y = (double *)malloc(n * d * sizeof(double));
  srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((double)rand() / (RAND_MAX)) * 100;

  return y;
}
__global__ void generateBoxIdx(uint64_t *Code, const double *Y, double scale,
                               const int nPts, const int nDim, const int nGrid,
                               const double multQuant, const uint32_t qLevel) {
  register uint64_t C[3];
  register double Yscale;
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

uint64_t tangleCode(const double *const YScat, const double scale,
                    const double multQuant, const uint32_t nGrid,
                    const uint32_t nDim) {

  uint32_t qLevel = ceil(log(nGrid) / log(2));

  uint64_t C[3];

  for (uint32_t j = 0; j < nDim; j++) {

    // get scaled input
    double Yscale = YScat[j] / scale;
    if (Yscale >= 1) {
      Yscale = 1 - std::numeric_limits<double>::epsilon();
      // printf("Yscale= %lf\n",Yscale );
    }

    // scale data points
    C[j] = (uint32_t)abs(floor(multQuant * Yscale));
  }

  switch (nDim) {

  case 1:
    return (uint64_t)C[0];

  case 2:
    return (((uint64_t)C[1]) << qLevel) | (((uint64_t)C[0]));

  case 3:
    return (((uint64_t)C[2]) << 2 * qLevel) | (((uint64_t)C[1]) << qLevel) |
           (((uint64_t)C[0]));

  default:
    return 0;
  }
}
void quantizeAndComputeCodes(uint64_t *const C, const double *const YScat,
                             const double scale, const uint32_t nPts,
                             const uint32_t nDim, const uint32_t nGrid) {

  // get quantization multiplier
  double multQuant = nGrid - 1 - std::numeric_limits<double>::epsilon();

  // add codes and ID to struct to sort them
  for (int i = 0; i < nPts; i++) {
    C[i] = tangleCode(&YScat[i * nDim], scale, multQuant, nGrid, nDim);
  }
}

template <typename dataval>
void doSort(uint64_t *const Cs, uint64_t *const Ct, uint32_t *const Ps,
            uint32_t *const Pt, dataval *const Ys, dataval *const Yt,
            uint32_t prev_off, const uint32_t nbits, const uint32_t sft,
            const uint32_t n, const uint32_t d, uint32_t nb) {

  // prepare bins
  uint32_t nBin = (0x01 << (nbits));
  // uint32_t *BinCursor  = new uint32_t[ nBin ]();
  uint32_t *BinCursor = (uint32_t *)calloc(nBin, sizeof(uint32_t));

  // current code
  uint32_t *code = new uint32_t[d]();

  // get mask for required number of bits
  uint64_t mask = (0x01 << (nbits)) - 1;

  for (int i = 0; i < n; i++) {
    uint32_t const ii = (Cs[i] >> sft) & mask;
    BinCursor[ii]++;
  }

  // scan prefix (can be better!)
  int offset = 0;
  for (int i = 0; i < nBin; i++) {
    int const ss = BinCursor[i];
    BinCursor[i] = offset;
    offset += ss;
  }

  // permute points
  for (int i = 0; i < n; i++) {
    uint32_t const ii = (Cs[i] >> sft) & mask;
    Ct[BinCursor[ii]] = Cs[i];
    for (int kapa = 0; kapa < d; kapa++) {
      Yt[BinCursor[ii] * d + kapa] = Ys[i * d + kapa];
    }
    Pt[BinCursor[ii]] = Ps[i];
    BinCursor[ii]++;
  }

  if (sft >= nbits) {

    offset = 0;
    for (int i = 0; i < nBin; i++) {
      uint32_t nPts = BinCursor[i] - offset;

      if (nPts > LIMIT_SEQ) {
        doSort(&Ct[offset], &Cs[offset], &Pt[offset], &Ps[offset],
               &Yt[offset * d], &Ys[offset * d], prev_off + offset, nbits,
               sft - nbits, nPts, d, nb);
      } else if (nPts > 0) {
        doSort(&Ct[offset], &Cs[offset], &Pt[offset], &Ps[offset],
               &Yt[offset * d], &Ys[offset * d], prev_off + offset, nbits,
               sft - nbits, nPts, d, nb);
      }
      offset = BinCursor[i];
    }
  }

  ;

  // delete BinCursor;
  free(BinCursor);
  delete code;
}

template <typename dataval>
void doSort_top(uint64_t *const Cs, uint64_t *const Ct, uint32_t *const Ps,
                uint32_t *const Pt, dataval *const Ys, dataval *const Yt,
                uint32_t prev_off, const uint32_t nbits, const uint32_t sft,
                const uint32_t n, const uint32_t d, uint32_t nb, uint32_t np) {

  // prepare bins
  uint32_t nBin = (0x01 << (nbits));

  // retrive active block per thread
  int m = (int)std::ceil((float)n / (float)np);

  uint32_t *BinCursor = (uint32_t *)calloc(nBin * np, sizeof(uint32_t));

  // current code
  uint32_t *code = new uint32_t[d]();

  // get mask for required number of bits
  uint64_t mask = (0x01 << (nbits)) - 1;

  for (int i = 0; i < np; i++) {
    int size = ((i + 1) * m < n) ? m : (n - i * m);
    for (int j = 0; j < size; j++) {
      uint32_t const ii = (Cs[i * m + j] >> sft) & mask;
      BinCursor[i * nBin + ii]++;
    }
  }

  int offset = 0;
  for (int i = 0; i < nBin; i++) {
    for (int j = 0; j < np; j++) {
      int const ss = BinCursor[j * nBin + i];
      BinCursor[j * nBin + i] = offset;
      offset += ss;
    }
  }

  // permute points
  for (int j = 0; j < np; j++) {
    int size = ((j + 1) * m < n) ? m : (n - j * m);
    for (int i = 0; i < size; i++) {
      uint32_t const idx = j * m + i;
      uint32_t const ii = (Cs[idx] >> sft) & mask;
      uint32_t const jj = BinCursor[j * nBin + ii];
      Ct[jj] = Cs[idx];
      for (int kapa = 0; kapa < d; kapa++) {
        Yt[jj * d + kapa] = Ys[idx * d + kapa];
      }
      Pt[jj] = Ps[idx];
      BinCursor[j * nBin + ii]++;
    }
  }

  if (sft >= nbits) {

    offset = 0;
    for (int i = 0; i < nBin; i++) {
      uint32_t nPts = BinCursor[(np - 1) * nBin + i] - offset;

      if (nPts > LIMIT_SEQ) {
        doSort(&Ct[offset], &Cs[offset], &Pt[offset], &Ps[offset],
               &Yt[offset * d], &Ys[offset * d], prev_off + offset, nbits,
               sft - nbits, nPts, d, nb);
      } else if (nPts > 0) {
        doSort(&Ct[offset], &Cs[offset], &Pt[offset], &Ps[offset],
               &Yt[offset * d], &Ys[offset * d], prev_off + offset, nbits,
               sft - nbits, nPts, d, nb);
      }
      offset = BinCursor[(np - 1) * nBin + i];
    }
  }

  // delete BinCursor;
  free(BinCursor);
  delete code;
}
__inline__ uint32_t untangleLastDim(const uint64_t C, const uint32_t nDim,
                                    const uint32_t qLevel) {

  uint32_t Cout = 0;

  switch (nDim) {

  case 1:
    Cout = (uint32_t)C;
    break;

  case 2: {
    uint64_t mask = (1 << 2 * qLevel) - 1;

    Cout = (uint32_t)((C & mask) >> qLevel);
    break;
  }

  case 3: {
    uint64_t mask = (1 << 3 * qLevel) - 1;

    Cout = (uint32_t)((C & mask) >> 2 * qLevel);
    break;
  }

  default: {
    std::cerr << "Supporting up to 3D" << std::endl;
    exit(1);
  }
  }

  return Cout;
}

void gridSizeAndIdx(uint32_t *const ib, uint32_t *const cb,
                    uint64_t const *const C, const uint32_t nPts,
                    const uint32_t nDim, const uint32_t nGridDim) {

  uint32_t qLevel = ceil(log(nGridDim) / log(2));
  uint32_t idxCur = -1;
  // printf("-----------------------Punch it mr sulu---------------------\n" );
  for (uint32_t i = 0; i < nPts; i++) {

    uint32_t idxNew = untangleLastDim(C[i], nDim, qLevel);
    // if(i<10){printf("idxNew=%d\n",idxNew );}
    cb[idxNew]++;

    if (idxNew != idxCur)
      ib[idxNew + 1] = i + 1;
  }
}
__inline__ __device__ uint32_t untangleLastDimDevice(int nDim, int TID,
                                                     uint32_t qLevel,
                                                     uint64_t *C) {
  uint64_t mask;
  switch (nDim) {
  case 1:
    return (uint32_t)C[TID];
    break;

  case 2: {
    mask = (1 << 2 * qLevel) - 1;

    return (uint32_t)((C[TID] & mask) >> qLevel);
    break;
  }

  case 3: {
    mask = (1 << 3 * qLevel) - 1;

    return (uint32_t)((C[TID] & mask) >> 2 * qLevel);
    break;
  }
  }
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

__global__ void ComputeChargesKernel(double *__restrict__ VScat,
                                     const double *const y_d, const int n,
                                     const int d, const int n_terms) {

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    for (int j = 0; j < d; j++) {
      VScat[TID + (j + 1) * n] = y_d[TID + (j)*n];
      // if(threadIdx.x==0){printf("y_d[%d]=%lf\n",TID+(j)*n ,y_d[TID+(j)*n]);}
    }
    VScat[TID] = 1;
  }
}
void ComputeCharges(double *VScat, double *y_d, int n, int d) {
  int threads = 1024;
  int Blocks = 64;
  ComputeChargesKernel<<<Blocks, threads>>>(VScat, y_d, n, d, d + 1);
}
__global__ void compute_repulsive_forces_kernel(
    volatile double *__restrict__ frep, const double *const Y,
    const int num_points, const int nDim, const double *const Phi,
    volatile double *__restrict__ zetaVec, uint32_t *iPerm) {
  register double Ysq = 0;
  register double z = 0;

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
double zetaAndForce(double *Ft_d, double *y_d, int n, int d, double *Phi,
                    thrust::device_vector<uint32_t> &iPerm,
                    thrust::device_vector<double> &zetaVec) {

  int threads = 1024;
  int Blocks = 64;
  compute_repulsive_forces_kernel<<<Blocks, threads>>>(
      Ft_d, y_d, n, d, Phi, thrust::raw_pointer_cast(zetaVec.data()),
      thrust::raw_pointer_cast(iPerm.data()));
  double z = thrust::reduce(zetaVec.begin(), zetaVec.end()) - n;
  return z;
}
template <typename dataval>
dataval zetaAndForce2(dataval *const F,            // Forces
                      const dataval *const Y,      // Coordinates
                      const dataval *const Phi,    // Values
                      const uint32_t *const iPerm, // Permutation
                      const uint32_t nPts,         // #points
                      const uint32_t nDim) {       // #dimensions

  dataval Z = 0;

  // compute normalization term
  for (uint32_t i = 0; i < nPts; i++) {
    dataval Ysq = 0;
    for (uint32_t j = 0; j < nDim; j++) {
      Ysq += Y[i * nDim + j] * Y[i * nDim + j];
      Z -= 2 * (Y[i * nDim + j] * Phi[i * (nDim + 1) + j + 1]);
    }
    Z += (1 + 2 * Ysq) * Phi[i * (nDim + 1)];
  }

  Z = Z - nPts;

  // Compute repulsive forces
  for (uint32_t i = 0; i < nPts; i++) {
    for (uint32_t j = 0; j < nDim; j++)
      F[iPerm[i] * nDim + j] = (Y[i * nDim + j] * Phi[i * (nDim + 1)] -
                                Phi[i * (nDim + 1) + j + 1]) /
                               Z;
  }

  return Z;
}

__global__ void s2g1dCuda(double *V, double *y, double *q, uint32_t ng,
                          uint32_t nPts, uint32_t nDim, uint32_t nVec,
                          double maxy);

void conv1dnopadcuda(double *PhiGrid_d, double *VGrid_d, double h, int nGridDim,
                     int m, int d);

void s2g1d(double *V, double *y, double *q, uint32_t ng, uint32_t np,
           uint32_t nPts, uint32_t nDim, uint32_t nVec);

void s2g1drb(double *V, double *y, double *q, uint32_t *ib, uint32_t *cb,
             uint32_t ng, uint32_t np, uint32_t nPts, uint32_t nDim,
             uint32_t nVec);
__global__ void g2s1dCuda(double *Phi, double *V, double *y, uint32_t ng,
                          uint32_t nPts, uint32_t nDim, uint32_t nVec);

void g2s1d(double *Phi, double *V, double *y, uint32_t ng, uint32_t nPts,
           uint32_t nDim, uint32_t nVec);

void nuconv(double *PhiScat, double *y, double *VScat, uint32_t *ib,
            uint32_t *cb, int n, int d, int m, int np, int nGridDim,
            double *y_d, uint32_t *id_d, uint32_t *cb_d, double *VScat_d,
            double *Phi_d) {

  // ~~~~~~~~~~ normalize coordinates (inside bins)
  double maxy = 0;
  int Blocks = 64;
  int threads = 1024;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for (int i = 0; i < n * d; i++) {
    y[i] /= maxy;
  }

  thrust::device_ptr<double> yVec_ptr(y_d);
  thrust::device_vector<double> yVec_d(yVec_ptr, yVec_ptr + n * d);
  thrust::device_vector<double>::iterator iter =
      thrust::max_element(yVec_d.begin(), yVec_d.end());
  unsigned int position = iter - yVec_d.begin();
  maxy = yVec_d[position];
  // ~~~~~~~~~~ scale them from 0 to ng-1

  for (int i = 0; i < n * d; i++) {
    if (y[i] == 1) {
      y[i] = y[i] - std::numeric_limits<double>::epsilon();
    }
    y[i] *= (nGridDim - 1);
  }

  for (int i = 0; i < n * d; i++)
    if ((y[i] >= nGridDim - 1) || (y[i] < 0))
      exit(1);

  // ~~~~~~~~~~ find exact h

  double h = maxy / (nGridDim - 1 - std::numeric_limits<double>::epsilon());

  // ~~~~~~~~~~ scat2grid
  int szV = pow(nGridDim + 2, d) * m;
  // printf("m=%d d=%d\n",m,d );
  double *VGrid = static_cast<double *>(calloc(szV * np, sizeof(double)));
  double *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d,
                              pow(nGridDim + 2, d) * (d + 1) * sizeof(double)));
  double *V = (double *)(calloc(szV * np, sizeof(double)));
  double *VScat2 = (double *)malloc(n * (d + 1) * sizeof(double));

  CUDA_CALL(cudaMemcpy(VScat2, VScat_d, n * (d + 1) * sizeof(double),
                       cudaMemcpyDeviceToHost));
  printf("====================================================================="
         "================\n");
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d + 1; j++) {
      if (abs(VScat[i * (d + 1) + j] - VScat2[i + j * n]) < 0.01) {
        // printf(" Succes VScat= %lf
        // VScat_d=%lf\n",VScat[i*(d+1)+j],VScat2[i+j*n] );
      } else {
        printf(" Error VScat= %lf VScat_d=%lf\n", VScat[i * (d + 1) + j],
               VScat2[i + j * n]);
      }
    }
  }
  printf("====================================================================="
         "==============\n");
  int tpoints = pow(nGridDim + 2, d);

  if (d == 1) {
    s2g1d(VGrid, y, VScat, nGridDim + 2, np, n, d, m);
    s2g1dCuda<<<1, 1>>>(VGrid_d, y_d, VScat_d, nGridDim + 2, n, d, m, maxy);
    // Differnt strategy every point must take 1 f1

    CUDA_CALL(
        cudaMemcpy(V, VGrid_d, szV * sizeof(double), cudaMemcpyDeviceToHost));

    for (int i = 0; i < pow(nGridDim + 2, d); i++) {
      for (int j = 0; j < m; j++) {
        if (abs(VGrid[i + j * tpoints] - V[i + j * tpoints]) < 0.00001) {
          // printf("Succes V1=%lf vs V=%lf \n",VGrid[i+j*tpoints] ,
          // V[i+j*tpoints]);
        } else {
          printf("Error V1=%lf vs V=%lf \n", VGrid[i + j * tpoints],
                 V[i + j * tpoints]);
        }
      }
    }

    printf("==================================================================="
           "================\n");
  }

  double *PhiGrid = static_cast<double *>(calloc(szV, sizeof(double)));
  double *PhiGrid_d;
  CUDA_CALL(cudaMallocManaged(&PhiGrid_d, szV * sizeof(double)));

  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = nGridDim + 2;
  }
  if (d == 1) {
    conv1dnopad(PhiGrid, VGrid, h, nGridDims, m, d, 1);
    conv1dnopadcuda(PhiGrid_d, VGrid_d, h, nGridDim + 2, m, d);
    double *PhiGrid_dtest = (double *)malloc(sizeof(double) * szV);

    CUDA_CALL(cudaMemcpy(PhiGrid_dtest, PhiGrid_d, szV * sizeof(double),
                         cudaMemcpyDeviceToHost));
    printf("==================================================================="
           "=\n");
    for (int i = 0; i < szV; i++) {
      printf("Vhost=%lf Vcuda=%lf\n", PhiGrid[i], PhiGrid_dtest[i]);
    }
    printf("==================================================================="
           "=\n");
    g2s1d(PhiScat, PhiGrid, y, nGridDim + 2, n, d, m);
    printf("======================================================\n");
    g2s1dCuda<<<32, 256>>>(Phi_d, PhiGrid_d, y_d, nGridDim + 2, n, d, m);
    double *phitest = (double *)malloc(m * n * sizeof(double));
    CUDA_CALL(cudaMemcpy(phitest, Phi_d, n * (d + 1) * sizeof(double),
                         cudaMemcpyDeviceToHost));
    for (int i = 0; i < n * m; i++) {
      // printf("Phih=%lf vs phicuda=%lf\n",PhiScat[i],phitest[i] );
    }
  }
}

__global__ void toZero(double *y, int j, double miny, int n) {

  for (register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < n;
       TID += gridDim.x * blockDim.x) {
    y[TID + j * n] = y[TID + j * n] - miny;
  }
}

int main(int argc, char **argv) {
  int d = atoi(argv[1]);
  int N = 1 << atoi(argv[2]);
  int threads = 1 << atoi(argv[3]);
  int blocks = 1 << atoi(argv[4]);
  int nGrid = atoi(argv[5]);

  double *y = generateRandomCoord(N, d);
  uint32_t *iPermpa = (uint32_t *)malloc(sizeof(uint32_t) * 1 * N);

  y[1] = 100;
  y[2] = 100;
  y[3] = 100;
  double *yc = (double *)malloc(N * d * sizeof(double));
  for (int i = 0; i < N; i++) {
    for (int j = 0; j < d; j++) {
      yc[i + N * j] = y[i * d + j];
      // printf("%lf  ",y[i*d+j] );
    }
    // printf("\n" );
  }
  double *y_d;
  int n = N;

  CUDA_CALL(cudaMallocManaged(&y_d, d * n * sizeof(double)));
  CUDA_CALL(
      cudaMemcpy(y_d, yc, n * d * sizeof(double), cudaMemcpyHostToDevice));

  thrust::device_ptr<double> yVec_ptr(y_d);
  thrust::device_vector<double> yVec_d(yVec_ptr, yVec_ptr + n * d);
  unsigned int position;
  double *miny = (double *)malloc(sizeof(double) * d);
  for (int j = 0; j < d; j++) {

    thrust::device_vector<double>::iterator iter = thrust::min_element(
        yVec_d.begin() + j * n, yVec_d.begin() + n * (j + 1));

    position = iter - (yVec_d.begin());

    miny[j] = yVec_d[position];
    // thrust::for_each(vec.begin()+j*n,vec.begin()+(j+1)*n , _1 -= miny[j]);
    toZero<<<blocks, threads>>>(y_d, j, miny[j], n);
    // printf("%lf\n",miny[j] );
  }
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      y[i * d + j] -= miny[j];
    }
  }

  thrust::device_vector<double>::iterator iter =
      thrust::max_element(yVec_d.begin(), yVec_d.end());
  position = iter - yVec_d.begin();
  double maxy = yVec_d[position];

  uint64_t *Codes;
  CUDA_CALL(cudaMallocManaged(&Codes, n * sizeof(uint64_t)));
  double multQuant = nGrid - 1 - std::numeric_limits<double>::epsilon();

  uint32_t qLevel = 0;
  qLevel = ceil(log(nGrid) / log(2));
  generateBoxIdx<<<blocks, threads>>>(Codes, y_d, 100, N, d, nGrid, multQuant,
                                      qLevel);
  uint64_t *Codes1;
  Codes1 = (uint64_t *)malloc(sizeof(uint64_t) * n);
  CUDA_CALL(
      cudaMemcpy(Codes1, Codes, sizeof(uint64_t) * n, cudaMemcpyDeviceToHost));

  uint64_t *Codes2;
  Codes2 = (uint64_t *)malloc(sizeof(uint64_t) * n);
  quantizeAndComputeCodes(Codes2, y, 100, n, d, nGrid);
  // printf("----------------------------------------\n" );
  for (int i = 0; i < n; i++) {
    if (Codes1[i] != Codes2[i]) {
      // printf("------------Error i=%d----------\n",i );}
    }

    cudaDeviceSynchronize(); // why

    thrust ::device_ptr<uint64_t> Codes_ptr(Codes);
    thrust::device_vector<uint32_t> iPerm(n);
    thrust::sequence(iPerm.begin(), iPerm.end());

    switch (d) {

    case 1:
      thrust ::stable_sort_by_key(
          Codes_ptr, Codes_ptr + n,
          make_zip_iterator(make_tuple(yVec_ptr, iPerm.begin())));

    case 2:
      thrust ::stable_sort_by_key(
          Codes_ptr, Codes_ptr + n,
          make_zip_iterator(make_tuple(yVec_ptr, yVec_ptr + n, iPerm.begin())));

    case 3:
      thrust ::stable_sort_by_key(
          Codes_ptr, Codes_ptr + n,
          make_zip_iterator(make_tuple(yVec_ptr, yVec_ptr + n, yVec_ptr + 2 * n,
                                       iPerm.begin())));
    }
    CUDA_CALL(cudaMemcpy(Codes1, Codes, sizeof(uint64_t) * n,
                         cudaMemcpyDeviceToHost));
    CUDA_CALL(
        cudaMemcpy(yc, y_d, sizeof(double) * d * n, cudaMemcpyDeviceToHost));
    uint32_t *perm = (uint32_t *)malloc(n * sizeof(uint32_t));
    CUDA_CALL(cudaMemcpy(perm, thrust::raw_pointer_cast(iPerm.data()),
                         sizeof(uint32_t) * n, cudaMemcpyDeviceToHost));
    /*
      for(int i=0;i<N;i++){
        printf("%ld ",Codes1[i] );
        for(int j=0;j<d;j++){

          printf("%lf  ",yc[i+N*j] );
        }
        printf(" --> %d ",perm[i] );
        printf("\n" );
      }*/
    uint64_t *const C2 = (uint64_t *)malloc(sizeof(uint64_t) * 1 * n);
    double *const Y2 = (double *)malloc(sizeof(double) * d * n);
    uint32_t *iPerm2 = (uint32_t *)malloc(sizeof(uint32_t) * 1 * n);
    for (int i = 0; i < n; i++) {
      iPermpa[i] = i;
    }
    doSort_top(Codes2, C2, iPermpa, iPerm2, y, Y2, 0, qLevel, (d - 1) * qLevel,
               n, d, nGrid, 1);

    // printf("----------------------------------------\n" );
    /*
      for(int i=0;i<n;i++){
        printf("C1=%ld C2=%ld  Y=",Codes2[i],C2[i]);
        for(int j=0;j<d;j++){

          printf("%lf  ",y[i*d+j] );
        }
        printf("Y2= " );
        for(int j=0;j<d;j++){

          printf("%lf  ",Y2[i*d+j] );
        }
        printf("\n" );
      }
    */
    uint32_t *ibh; // Starting index of box (along last dimension)
    uint32_t *cbh; // Number of scattered points per box (along last dimension)
    ibh = (uint32_t *)calloc(nGrid, sizeof(uint32_t));
    cbh = (uint32_t *)calloc(nGrid, sizeof(uint32_t));
    uint32_t *ib;
    uint32_t *cb;
    CUDA_CALL(cudaMallocManaged(&ib, nGrid * sizeof(uint32_t)));
    CUDA_CALL(
        cudaMemcpy(ib, ibh, nGrid * sizeof(uint32_t), cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMallocManaged(&cb, nGrid * sizeof(uint32_t)));
    CUDA_CALL(
        cudaMemcpy(cb, cbh, nGrid * sizeof(uint32_t), cudaMemcpyHostToDevice));
    if ((d % 2) == 1) {

      // ========== get starting index and size of each grid box
      gridSizeAndIdx(ibh, cbh, C2, n, d, nGrid);
      y = Y2;
      iPermpa = iPerm2;

    } else {

      // ========== get starting index and size of each grid box
      gridSizeAndIdx(ibh, cbh, Codes2, n, d, nGrid);
    }

    // printf("------------Please god help me-------\n" );

    for (int i = 0; i < n; i++) {
      if (iPermpa[i] != perm[i]) {
        printf("Error %d   vs %d\n", iPermpa[i], perm[i]);
      }
    }

    gridSizeAndIdxKernel<<<blocks, threads>>>(ib, cb, Codes, n, d, nGrid,
                                              qLevel);
    uint32_t *ib2 = (uint32_t *)calloc(nGrid, sizeof(uint32_t));

    CUDA_CALL(
        cudaMemcpy(ib2, ib, nGrid * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (int i = 0; i < nGrid; i++) {
      if (ib2[i] != ibh[i]) {
        printf("Error ib=%d ibh=%d \n", ib2[i], ibh[i]);
      } else {
        printf("Succes ib=%d ibh=%d \n", ib2[i], ibh[i]);
      }
    }
    uint32_t *cb2 = (uint32_t *)calloc(nGrid, sizeof(uint32_t));
    CUDA_CALL(
        cudaMemcpy(cb2, cb, nGrid * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < nGrid; i++) {
      if (cbh[i] != cb2[i]) {
        printf("Error cbh=%d cb=%d\n", cbh[i], cb2[i]);
      } else {
        printf("cudaSuccess cbh=%d cb=%d\n", cbh[i], cb2[i]);
      }
    }

    double *VScat = (double *)malloc(n * (d + 1) * sizeof(double));
    double *VScat_d;
    CUDA_CALL(cudaMallocManaged(&VScat_d, n * (d + 1) * sizeof(double)));
    ComputeCharges(VScat_d, y_d, n, d);
    double *VScat2 = (double *)malloc(n * (d + 1) * sizeof(double));

    CUDA_CALL(cudaMemcpy(VScat2, VScat_d, n * (d + 1) * sizeof(double),
                         cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {

      VScat[i * (d + 1)] = 1.0;
      for (int j = 0; j < d; j++)
        VScat[i * (d + 1) + j + 1] = y[i * d + j];
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < d + 1; j++) {
        if (VScat[i * (d + 1) + j] == VScat2[i + j * n]) {
          // printf(" Succes VScat= %lf
          // VScat_d=%lf\n",VScat[i*(d+1)+j],VScat2[i+j*n] );
        } else {
          printf(" Error VScat= %lf VScat_d=%lf\n", VScat[i * (d + 1) + j],
                 VScat2[i + j * n]);
        }
      }
    }

    for (int i = 0; i < n; i++) {
      for (int j = 0; j < d; j++)
        if (y[i * (d) + j] == yc[i + j * n]) {
          //   printf(" Succes y= %lf yc=%lf\n",y[i*(d)+j],yc[i+j*n] );
        } else {
          printf(" Error y= %lf yc=%lf\n", y[i * (d) + j], yc[i + j * n]);
        }
    }

    double *Phi = generateRandomCoord(n, d + 1);
    double *Phi_d;
    CUDA_CALL(cudaMallocManaged(&Phi_d, n * (d + 1) * sizeof(double)));
    nuconv(Phi, y, VScat, ibh, cbh, n, d, d + 1, 1, nGrid, y_d, ib, cb, VScat_d,
           Phi_d);

    // thrust::device_vector<double> zetaVec(n);
    // double* Ft=(double *)malloc(d*n*sizeof(double));
    // double* Ft_d;
    // CUDA_CALL(cudaMallocManaged(&Ft_d,n*d * sizeof( double )));
    // CUDA_CALL(cudaMemcpy(Phi_d,Phi,n*(d+1) * sizeof( double ),
    // cudaMemcpyHostToDevice));

    /*
      double z= zetaAndForce( Ft_d, y_d, n, d, Phi_d,iPerm,zetaVec);
      double z2=zetaAndForce2(Ft,y,Phi,iPermpa,n,d);
      printf("%lf vs %lf \n",z,z2 );
      double* Ft2=(double *)malloc(d*n*sizeof(double));
      CUDA_CALL(cudaMemcpy(Ft2,Ft_d,n*(d) * sizeof( double ),
      cudaMemcpyDeviceToHost)); for(int i=0;i<n;i++){ for(int j=0;j<d;j++){
          if(abs(Ft[i*d+j]-Ft2[i+j*n]/z)>0.0000001){
          printf("Error F1=%lf vs Fd=%lf \n",Ft[i*d+j],Ft2[i+j*n]/z );}
        }
      }
    */

    return 0;
  }
