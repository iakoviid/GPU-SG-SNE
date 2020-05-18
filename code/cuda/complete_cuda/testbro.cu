#include "gridding.cuh"
#include "gridding.hpp"
#include "relocateData.cuh"
#include "relocateData.hpp"
#include "non_periodic_conv.cuh"
#include "non_periodic_conv.hpp"

#include <iostream>
using namespace std;
typedef double coord;

coord *generateRandomCoord(int n, int d) {

  coord *y = (coord *)malloc(n * d * sizeof(coord));
  srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((coord)rand() / (RAND_MAX)) * 100;

  return y;
}

template <class dataPoint>
void copydata(dataPoint *const w, dataPoint *dw, int n, int d) {
  dataPoint *v = (dataPoint *)malloc(sizeof(dataPoint) * n * d);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {

      v[i + n * j] = w[i * d + j];
    }
  }
  cudaMemcpy(dw, v, d * n * sizeof(dataPoint), cudaMemcpyHostToDevice);
  free(v);
  return;
}

template <class dataPoint>
void compair(dataPoint *const w, dataPoint *dv, int n, int d, char *message,
             int same) {
  int bro = 1;
  dataPoint *v = (dataPoint *)malloc(n * d * sizeof(dataPoint));
  cudaMemcpy(v, dv, d * n * sizeof(dataPoint), cudaMemcpyDeviceToHost);
  printf(
      "----------------------------------Compair %s----------------------------"
      "-----------\n",
      message);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      if (same == 0) {
        if (abs(w[i * d + j] - v[i + j * n]) < 0.1 * abs(w[i * d + j])) {
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else {
          bro = 0;
          cout << "Error "
               << "Host=" << w[i * d + j] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      } else {
        if (abs(w[i + j * n] - v[i + j * n]) < 0.1 * abs(w[i + j * n])) {
          // printf("Succes host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        } else {
          bro = 0;
          cout << "Error "
               << "Host=" << w[i + j * n] << " vs Cuda=" << v[i + j * n]
               << "in position i=" << i << " n=" << n << endl;
          // printf("Error host=%lf vs cuda=%lf\n",w[i*d+j],v[i+j*n]);
        }
      }
    }
  }
  if (bro == 1) {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Success~~~~~~~~~~~~~~~~~~~~~~~~\n");
  } else {
    printf(
        "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~Error~~~~~~~~~~~~~~~~~~~~~~~~\n");
  }
  free(v);
}

int main(int argc, char **argv) {
  int n = 1 << atoi(argv[1]);
  int d = atoi(argv[2]);
  int ng = atoi(argv[3]);
  coord *y, *y_d;
  CUDA_CALL(cudaMallocManaged(&y_d, (d)*n * sizeof(coord)));
  thrust::device_vector<uint32_t> iPerm(n);
  thrust::sequence(iPerm.begin(), iPerm.end());
  uint32_t *iPerm_h = (uint32_t *)malloc(n * sizeof(uint32_t));
  for (int i = 0; i < n; i++) {
    iPerm_h[i] = i;
  }
  y = generateRandomCoord(n, d);
  copydata(y, y_d, n, d);
  uint32_t *ib, *cb, *ib_h, *cb_h;
  CUDA_CALL(cudaMallocManaged(&ib, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&ib_h, ng * sizeof(uint32_t)));
  CUDA_CALL(cudaMallocManaged(&cb_h, ng * sizeof(uint32_t)));
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);

  relocateCoarseGrid(y_d, iPerm, ib, cb, n, ng, d);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Relocation 1 milliseconds %f\n", milliseconds);

  cudaEventRecord(start);

  relocateCoarseGridCPU(&y, &iPerm_h, ib_h, cb_h, n, ng, d, 1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Relocation 1 milliseconds %f\n", milliseconds);

  int szV = pow(ng + 2, d) * (d + 1);

  coord *VGrid = (coord *)calloc(szV, sizeof(coord));
  coord *VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV * sizeof(coord)));

  coord *VScat = generateRandomCoord(n, d + 1);
  coord *VScat_d;
  CUDA_CALL(cudaMallocManaged(&VScat_d, (d + 1) * n * sizeof(coord)));
  copydata(VScat, VScat_d, n, d + 1);

  cudaEventRecord(start);
  coord maxy = 0;
  for (int i = 0; i < n * d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for (int i = 0; i < n * d; i++) {
    y[i] /= maxy;

    // ~~~~~~~~~~ scale them from 0 to ng-1

    if (1 == y[i]) {
      y[i] = y[i] - std::numeric_limits<coord>::epsilon();
    }

    y[i] *= (ng - 1);
  }
  coord h = maxy / (ng - 1 - std::numeric_limits<double>::epsilon());

  s2g1dCpu(VGrid, y, VScat, ng + 2, 1, n, d, d + 1);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Transform 1 milliseconds %f\n", milliseconds);

  cudaEventRecord(start);

  s2g1d<<<32, 512>>>(VGrid_d, y_d, VScat_d, ng + 2, n, d, d + 1, maxy);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("Transform 1 milliseconds %f\n", milliseconds);

  compair(y, y_d, n, d, "Y", 0);
  compair(VGrid, VGrid_d, pow(ng + 2, d), d + 1, "VGrid", 1);
  compair(iPerm_h, thrust::raw_pointer_cast(iPerm.data()), n,1, "iPerm", 0);
  int nVec=d+1;
  coord* Phi=(coord *)malloc(n*nVec*sizeof(coord));
  coord* Phi_d;
  CUDA_CALL(cudaMallocManaged(&Phi_d,nVec*n*sizeof(coord)));

  g2s1dCpu(Phi,VGrid,y,ng,n,d,nVec);
  g2s1d<<<32,256>>>(Phi_d,VGrid_d,y_d,ng,n,d,nVec);

  compair(Phi, Phi_d, n, d + 1, "Phi", 0);
  uint32_t m=d+1;

  uint32_t *const nGridDims = new uint32_t[d]();
  for (int i = 0; i < d; i++) {
    nGridDims[i] = ng + 2;
  }
  coord *PhiGrid = static_cast<coord *>(calloc(szV, sizeof(coord)));
  coord *PhiGrid_d;
  CUDA_CALL(cudaMallocManaged(&PhiGrid_d, szV * sizeof(coord)));

  conv1dnopad(PhiGrid, VGrid, h, nGridDims, m, d, 1);
  conv1dnopadcuda(PhiGrid_d, VGrid_d, h, ng + 2, m, d);
  uint32_t tpoints = pow(ng + 2, d);


  compair(PhiGrid, PhiGrid_d, tpoints, d + 1, "PhiFFT", 0);


  return 0;
}
