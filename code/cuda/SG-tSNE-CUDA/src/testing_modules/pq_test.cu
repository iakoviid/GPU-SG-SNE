#include "../bsr_pq.cuh"
#include "../matrix_converter.h"
#include "../types.hpp"
#include <algorithm>
#include <cusparse.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sys/time.h>
#include <unordered_map>
#include "../hybrid.cuh"
#include "../prepareMatrix.cuh"


#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }
using namespace std;

float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
  return dis(e);
}
int get_random_int(int max) {
  static std::default_random_engine e;
  std::uniform_int_distribution<int> dis(0, max - 1);
  return dis(e);
}

void generate_candidate_blocks(int R, int C, int BS_R, int BS_C, int num_blocks,
                               int *weight_indptr, int *weight_indices) {
  std::map<int, std::set<int>> blocks;
  int num_r_block = R;
  int num_c_block = C;
  int curr_size = 0;
  while (curr_size < num_blocks) {
    int r = get_random_int(num_r_block);
    int c = get_random_int(num_c_block);
    if (blocks[r].count(c) == 0) {
      blocks[r].insert(c);
      curr_size++;
    }
  }

  int current_ptr = 0;
  int i;
  for (i = 0; i < num_r_block; i++) {
    weight_indptr[i] = current_ptr;
    for (auto block : blocks[i]) {
      weight_indices[current_ptr++] = block;
    }
  }
  weight_indptr[i] = current_ptr;
}
__global__ void ComputePijxQijKernel(coord *__restrict__ attr_forces,
                                     const coord *__restrict__ pij,
                                     const coord *__restrict__ points,
                                     const int *__restrict__ coo_indicesi,
                                     const int *__restrict__ coo_indicesj,
                                     const int num_points,
                                     const int num_nonzero) {
  register int TID, i, j;
  register coord ix, iy, jx, jy, dx, dy, pijqij;
  TID = threadIdx.x + blockIdx.x * blockDim.x;
  if (TID >= num_nonzero)
    return;
  i = coo_indicesi[TID];
  j = coo_indicesj[TID];

  ix = points[i];
  iy = points[num_points + i];
  jx = points[j];
  jy = points[num_points + j];
  dx = ix - jx;
  dy = iy - jy;
  pijqij = pij[TID] / (1 + dx * dx + dy * dy);
  atomicAdd(attr_forces + i, pijqij * dx);
  atomicAdd(attr_forces + num_points + i, pijqij * dy);
}
__host__ __device__ int iDivUp(int a, int b) {
  return ((a % b) != 0) ? (a / b + 1) : (a / b);
}

double tsne_cuda_spmv(coord *Fattr, coord *coov, coord *Y, int *cooi, int *cooj,
                      int n, int nnz, int d) {
  coord *Fattr_d, *Y_d, *coov_d;
  int *cooi_d, *cooj_d;
  CUDA_CALL(cudaMalloc(&Fattr_d, n * d * sizeof(coord)));
  CUDA_CALL(cudaMemcpy(Fattr_d, Fattr, n * d * sizeof(coord),
                       cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&Y_d, n * d * sizeof(coord)));
  CUDA_CALL(cudaMemcpy(Y_d, Y, n * d * sizeof(coord), cudaMemcpyHostToDevice));

  CUDA_CALL(cudaMalloc(&cooi_d, nnz * sizeof(int)));
  CUDA_CALL(
      cudaMemcpy(cooi_d, cooi, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cooj_d, nnz * sizeof(int)));
  CUDA_CALL(
      cudaMemcpy(cooj_d, cooj, nnz * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&coov_d, nnz * sizeof(coord)));
  CUDA_CALL(
      cudaMemcpy(coov_d, coov, nnz * sizeof(coord), cudaMemcpyHostToDevice));

  const int BLOCKSIZE = 1024;
  const int NBLOCKS = iDivUp(nnz, BLOCKSIZE);
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);

  ComputePijxQijKernel<<<NBLOCKS, BLOCKSIZE>>>(Fattr_d, coov_d, Y_d, cooi_d,
                                               cooj_d, n, nnz);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  //  printf("COO elapsedTime=%lf\n", elapsedTime);
  CUDA_CALL(cudaMemcpy(Fattr, Fattr_d, n * d * sizeof(coord),
                       cudaMemcpyDeviceToHost));

  cudaFree(Fattr_d);
  cudaFree(Y_d);
  cudaFree(cooi_d);
  cudaFree(cooj_d);
  cudaFree(coov_d);
  return elapsedTime;
}
template <class dataPoint>
dataPoint maxerror(dataPoint *const w, dataPoint *v, int n, int d) {
  printf("Enter Comparison_-----------------------------\n" );
  dataPoint maxError = 0;
  dataPoint avgError = 0;
  int pos = 0;

  for (int i = 0; i < n; i++) {
    for (int j = 0; j < d; j++) {
      /*
      if((v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]) >
          0.001 || i<2){
      cout<<"v= "<<v[i + j * n]<< "w= "<<w[i + j * n]<<"\n";
    }*/

      if ((v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]) >
          maxError) {
        maxError =
            (v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]);
        pos = i;
      }
      avgError += (v[i + j * n] - w[i + j * n]) * (v[i + j * n] - w[i + j * n]);
    }
  }

  // printf("maxError=%lf pos=%d v[i]=%lf vs w[i]=%lf avgError=%lf n=%d
  // size=%d\n",
  //         maxError, pos, v[pos], w[pos], avgError / (n * d), n, n * d);

  return maxError;
}
double test_coo(coord *coov, int *cooi, int *cooj, int nnz, coord *Y, int n,
                int d, coord *Fserial) {
  coord *Fcoo = (coord *)calloc(sizeof(coord), n * d);

  double time = tsne_cuda_spmv(Fcoo, coov, Y, cooi, cooj, n, nnz, d);
  double maxError = maxerror(Fserial, Fcoo, n, d);
  printf("COO maxError=%lf\n",maxError );
  if (maxError > 0.00001) {
    printf("Error coo\n");
  }
  free(Fcoo);
  return time;
}

void serial(coord *val, int *block_ptr, int *col_ind, coord *Y, coord *Fattr,
            int n, int bs, int d, int m) {
  for (int i = 0; i < n; i++) {
    int block_first = block_ptr[i];
    int block_last = block_ptr[i + 1];
    for (int block = block_first; block < block_last; block++) {
      for (int row = 0; row < bs; row++) {
        for (int col = 0; col < bs; col++) {
          // printf("%d  %d  %f \n", i * bs + row,
          int row_v = i * bs + row;

          int column = col_ind[block] * bs + col;
          coord dist = 0;
          for (int dim = 0; dim < d; dim++) {
            dist += (Y[row_v + dim * m] - Y[column + dim * m]) *
                    (Y[row_v + dim * m] - Y[column + dim * m]);
          }
          for (int dim = 0; dim < d; dim++) {
            Fattr[row_v + m * dim] +=
                val[block * bs * bs + row * bs + col] *
                (Y[row_v + m * dim] - Y[column + m * dim]) / (dist + 1);
          }
        }
      }
    }
  }
}

double bsr1run(coord *val, int *block_ptr, int *col_ind, coord *y, coord *Fattr,
               int n, int blockSize, int num_blocks, int rows, int d) {
  coord *cu_y;
  coord *cu_F;
  CUDA_CALL(cudaMalloc(&cu_y, rows * d * sizeof(coord)));
  CUDA_CALL(
      cudaMemcpy(cu_y, y, rows * d * sizeof(coord), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cu_F, rows * d * sizeof(coord)));
  CUDA_CALL(cudaMemcpy(cu_F, Fattr, rows * d * sizeof(coord),
                       cudaMemcpyHostToDevice));
  coord *val_d;
  int *col_ind_d, *block_ptr_d;
  dim3 block_size = 512;
  dim3 grid_size{};
  grid_size.x = (n * blockSize + block_size.x - 1) / block_size.x;

  cudaMalloc(&val_d, num_blocks * blockSize * blockSize * sizeof(coord));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(int));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(int));
  cudaMemcpy(val_d, val, num_blocks * blockSize * blockSize * sizeof(coord),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  bsrpq_r<coord, int><<<grid_size, block_size>>>(
      n, blockSize, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  //  printf("BSR row elapsedTime=%lf\n", elapsedTime);
  cudaMemcpy(Fattr, cu_F, rows * d * sizeof(coord), cudaMemcpyDeviceToHost);

  CUDA_CALL(cudaFree(cu_F));
  CUDA_CALL(cudaFree(cu_y));
  CUDA_CALL(cudaFree(val_d));
  CUDA_CALL(cudaFree(col_ind_d));
  CUDA_CALL(cudaFree(block_ptr_d));
  return elapsedTime;
}
template <typename data_type, typename index_type>
void transpose_blocks(data_type *new_values, index_type *row_ptr, index_type bs,
                      index_type n_rows, data_type *values) {
  std::unique_ptr<data_type[]> buffer(new data_type[bs * bs]);

  for (index_type row = 0; row < n_rows; row++) {
    for (index_type block = row_ptr[row]; block < row_ptr[row + 1]; block++) {
      data_type *new_block_data = new_values + bs * bs * block;
      data_type *old_block_data = values + bs * bs * block;
      std::copy_n(old_block_data, bs * bs, buffer.get());

      for (unsigned int i = 0; i < bs; i++)
        for (unsigned int j = 0; j < bs; j++)
          new_block_data[j * bs + i] = buffer[i * bs + j];
    }
  }
}
double bsr_colrun(coord *val, int *block_ptr, int *col_ind, coord *y,
                  coord *Fattr, int n, int blockSize, int num_blocks, int rows,
                  int d) {

  coord *cu_y;
  coord *cu_F;
  CUDA_CALL(cudaMalloc(&cu_y, rows * d * sizeof(coord)));
  CUDA_CALL(
      cudaMemcpy(cu_y, y, rows * d * sizeof(coord), cudaMemcpyHostToDevice));
  CUDA_CALL(cudaMalloc(&cu_F, rows * d * sizeof(coord)));
  CUDA_CALL(cudaMemcpy(cu_F, Fattr, rows * d * sizeof(coord),
                       cudaMemcpyHostToDevice));
  coord *val_t =
      (coord *)malloc(sizeof(coord) * num_blocks * blockSize * blockSize);
  transpose_blocks<coord, int>(val_t, block_ptr, blockSize, n, val);

  coord *val_d;
  int *col_ind_d, *block_ptr_d;
  cudaMalloc(&val_d, num_blocks * blockSize * blockSize * sizeof(coord));
  cudaMalloc(&block_ptr_d, (n + 1) * sizeof(int));
  cudaMalloc(&col_ind_d, num_blocks * sizeof(int));
  cudaMemcpy(val_d, val_t, num_blocks * blockSize * blockSize * sizeof(coord),
             cudaMemcpyHostToDevice);
  cudaMemcpy(block_ptr_d, block_ptr, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(col_ind_d, col_ind, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);

  dim3 block_size = 32;
  dim3 grid_size{};

  grid_size.x = (n * 32 + block_size.x - 1) / block_size.x;
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  switch (blockSize) {
  case 1:
    bsr_col<coord, int, 1>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 2:
    bsr_col<coord, int, 2>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 3:
    bsr_col<coord, int, 3>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 4:
    bsr_col<coord, int, 4>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 8:
    bsr_col<coord, int, 8>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 16:
    bsr_col<coord, int, 16>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  case 32:
    bsr_col<coord, int, 32>
        <<<grid_size, block_size, block_size.x * d * sizeof(coord)>>>(
            n, col_ind_d, block_ptr_d, val_d, cu_y, cu_F, rows, d);
    break;
  }
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  //  printf("BSR col elapsedTime=%lf\n",elapsedTime );

  cudaMemcpy(Fattr, cu_F, rows * d * sizeof(coord), cudaMemcpyDeviceToHost);
  CUDA_CALL(cudaFree(cu_F));
  CUDA_CALL(cudaFree(cu_y));
  CUDA_CALL(cudaFree(val_d));
  CUDA_CALL(cudaFree(col_ind_d));
  CUDA_CALL(cudaFree(block_ptr_d));
  return elapsedTime;
}
void test_pq(coord *val, int *block_ptr, int *col_ind, coord *y, int n,
             int blockSize, int num_blocks, int rows, int d, coord *Fserial,
             double *timeInfo) {
  coord *Fbsr1 = (coord *)calloc(sizeof(coord), rows * d);
  timeInfo[0] = bsr1run(val, block_ptr, col_ind, y, Fbsr1, n, blockSize,
                        num_blocks, rows, d);

  double maxError = maxerror(Fserial, Fbsr1, rows, d);
  printf("BSR1 maxError=%lf\n",maxError );
  if (maxError > 0.000001) {
    printf("Error Bsr 1\n");
  }
  coord *Fbsr2 = (coord *)calloc(sizeof(coord), rows * d);
  timeInfo[1] = bsr_colrun(val, block_ptr, col_ind, y, Fbsr2, n, blockSize,
                           num_blocks, rows, d);
  maxError = maxerror(Fserial, Fbsr2, rows, d);
  printf("BSR2 maxError=%lf\n",maxError );
  if (maxError > 0.00001) {
    printf("Error Bsr 2\n");
  }

  free(Fbsr1);
  free(Fbsr2);
}
void bsr2csr(coord *bsrValA, int *bsrRowPtrA, int *bsrColIndA, int n, int bs,
             int num_blocks, int rows, coord *csrVal, int *csrRptr,
             int *csrCidx) {
  // Given BSR format (bsrRowPtrA, bsrcolIndA, bsrValA) and
  // blocks of BSR format are stored in column-major order.
  cusparseDirection_t dir = CUSPARSE_DIRECTION_ROW;
  int m = n * bs;
  int nnz = num_blocks * bs * bs; // number of elements
  int *csrRowPtrC, *csrColIndC;
  coord *csrValC;
  cudaMalloc((void **)&csrRowPtrC, sizeof(int) * (m + 1));
  cudaMalloc((void **)&csrColIndC, sizeof(int) * nnz);
  cudaMalloc((void **)&csrValC, sizeof(coord) * nnz);
  coord *bsrValA_d;
  int *bsrColIndA_d, *bsrRowPtrA_d;
  cudaMalloc(&bsrValA_d, nnz * sizeof(coord));
  cudaMalloc(&bsrRowPtrA_d, (n + 1) * sizeof(int));
  cudaMalloc(&bsrColIndA_d, num_blocks * sizeof(int));
  cudaMemcpy(bsrValA_d, bsrValA, nnz * sizeof(coord), cudaMemcpyHostToDevice);
  cudaMemcpy(bsrRowPtrA_d, bsrRowPtrA, (n + 1) * sizeof(int),
             cudaMemcpyHostToDevice);
  cudaMemcpy(bsrColIndA_d, bsrColIndA, num_blocks * sizeof(int),
             cudaMemcpyHostToDevice);
  cusparseMatDescr_t descr = 0;
  cusparseCreateMatDescr(&descr);
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  cusparseDbsr2csr(handle, dir, n, n, descr, bsrValA_d, bsrRowPtrA_d,
                   bsrColIndA_d, bs, descr, csrValC, csrRowPtrC, csrColIndC);
  cudaMemcpy(csrVal, csrValC, nnz * sizeof(coord), cudaMemcpyDeviceToHost);
  cudaMemcpy(csrCidx, csrColIndC, nnz * sizeof(int), cudaMemcpyDeviceToHost);
  cudaMemcpy(csrRptr, csrRowPtrC, (rows + 1) * sizeof(int),
             cudaMemcpyDeviceToHost);

  cudaFree(bsrValA_d);
  cudaFree(bsrColIndA_d);
  cudaFree(bsrRowPtrA_d);
  cudaFree(csrRowPtrC);
  cudaFree(csrColIndC);
  cudaFree(csrValC);
}
/*
void loadblocks(int N,int bs,int num_blocks,int weight_ptr,int  weight_ind,coord weight){
  for(int i=0; i<mb+1;i++){
    scanf("%d\n", &weight_ptr[i]);

  }
  for(int i=0;i<num_blocks*bs*bs;i++){
    scanf("%lf\n", &weight[i]);

  }
  for(int i=0;i<num_blocks;i++)
  {
    scanf("%d\n", &weight_ind[i]);

  }
}
*/

int main(int argc, char **argv) {
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);

  ofstream myfile;
  ofstream myfile2;
  myfile.open("matrix.txt");
  myfile2.open("Y.txt");
  int N = 1 << atoi(argv[1]);
  int K = N;
  int bs = 1 << atoi(argv[2]);
  myfile << "N= " << N << " bs= " << bs << "\n";
  float density = (float)1 / (1 << (atoi(argv[3])));
  int d = atoi(argv[4]);
  int iterations = atoi(argv[5]);
  cout << "N= " << N << " d= " << d << " bs= " << bs << " density= " << density
       << "\n";
  /*Make random bsr matrix N*bs x K*bs */
  coord *weight;
  int *weight_ind;
  int *weight_ptr;
  int nnz = int(density * K * N * bs * bs); // nnz
  int num_blocks = int(nnz / (bs * bs)) + 1;
  printf("num_blocks=%d\n", num_blocks); // will be generated from blocks
  nnz = num_blocks * bs * bs;
  printf("nnz=%d\n", nnz);

  weight = (coord *)malloc(num_blocks * bs * bs * sizeof(coord));
  weight_ind = (int *)malloc(num_blocks * sizeof(int));
  weight_ptr = (int *)malloc((N + 1) * sizeof(int));
  //loadblocks(N,bs,num_blocks,weight_ptr, weight_ind,weight);
  for (int i = 0; i < num_blocks * bs * bs; i++) {
    weight[i] = (coord)10 * get_random();
  }
  generate_candidate_blocks(N, K, bs, bs, num_blocks, weight_ptr, weight_ind);
  int rows = N * bs;
  coord *x = (coord *)malloc(rows * d * sizeof(coord));
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < d; j++) {
      x[i + j * rows] = (coord)10 * get_random();
      myfile2 << x[i + j * rows] << " ";
    }
    myfile2 << "\n";
  }
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("Making elapsedTime=%lf\n", elapsedTime);
  // Make coo format

  coord *coov = (coord *)malloc(sizeof(coord) * nnz);
  int *cooi = (int *)malloc(sizeof(int) * nnz);
  int *cooj = (int *)malloc(sizeof(int) * nnz);
  int nnzcntr = 0;
  gettimeofday(&t1, NULL);

  for (int i = 0; i < N; i++) {
    int block_first = weight_ptr[i];
    int block_last = weight_ptr[i + 1];
    for (int block = block_first; block < block_last; block++) {
      for (int row = 0; row < bs; row++) {
        for (int col = 0; col < bs; col++) {

          myfile << i * bs + row << " " << weight_ind[block] * bs + col << " "
                 << weight[block * bs * bs + row * bs + col] << "\n";

          coov[nnzcntr] = weight[block * bs * bs + row * bs + col];
          cooi[nnzcntr] = i * bs + row;
          cooj[nnzcntr] = weight_ind[block] * bs + col;

          nnzcntr++;
          // bsr[ i * bs + row][weight_ind[block] * bs + col]=weight[block * bs
          // * bs + row * bs + col];
        }
      }
    }
  }
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("Coo and file elapsedTime=%lf\n", elapsedTime);

  myfile.close();
  myfile2.close();
  coord *Fserial = (coord *)calloc(sizeof(coord), rows * d);

  gettimeofday(&t1, NULL);
  serial(weight, weight_ptr, weight_ind, x, Fserial, N, bs, d, rows);
  gettimeofday(&t2, NULL);
  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  printf("serial  elapsedTime=%lf\n", elapsedTime);
  double timeInfo[2];
  double *time_coo = (double *)malloc(iterations * sizeof(double));
  double *time_bsr_row = (double *)malloc(iterations * sizeof(double));
  double *time_bsr_col = (double *)malloc(iterations * sizeof(double));


  coord *csrVal = (coord *)malloc(sizeof(coord) * nnz);
  int *csrCidx = (int *)malloc(sizeof(int) * nnz);
  int *csrRptr = (int *)malloc(sizeof(int) * (rows + 1));
  bsr2csr(weight, weight_ptr, weight_ind, N, bs, num_blocks, rows, csrVal,
          csrRptr, csrCidx);
  coord* bsrValC;
  int* bsrRowPtrC,*bsrColIndC;
  int mb,nnzb;
  cusparseHandle_t handle;
  cusparseCreate(&handle);
  csr2bsr( bs, rows, rows,  nnz,csrRptr,csrCidx,csrVal, &bsrRowPtrC, &bsrColIndC, &bsrValC,&nnzb,&mb,handle);

  coord* bsrValh=(coord*)malloc(sizeof(coord)*nnzb*bs*bs );
  int* bsrColh=(int*)malloc(sizeof(int)*nnzb);
  int* bsrRowh=(int* )malloc(sizeof(int)*(mb+1));

  CUDA_CALL(cudaMemcpy(bsrRowh,bsrRowPtrC , (mb+1) *sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(bsrColh, bsrColIndC,  nnzb*sizeof(int),cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(bsrValh, bsrValC,  bs*bs*nnzb*sizeof(coord),cudaMemcpyDeviceToHost));

  printf("Index: %d\n",maxerror(bsrColh,weight_ind, nnzb, 1) );
  printf("Index: %lf\n",maxerror(bsrValh,weight, nnzb*bs*bs, 1) );
  printf("Index: %d\n",maxerror(bsrRowh,weight_ptr, N+1, 1) );
  printf("nnzb=%d mb=%d\n",nnzb,mb );
  printf("num_blocks=%d N=%d\n", num_blocks,N); // will be generated from blocks
  coord* hybridy;
  coord* dx;
  CUDA_CALL(cudaMalloc(&hybridy, rows*d * sizeof(coord)));
  CUDA_CALL(cudaMalloc(&dx, rows*d * sizeof(coord)));
  cudaMemcpy(dx, x, rows*d * sizeof (coord),cudaMemcpyHostToDevice);


        double *time_hybrid = (double *)malloc(iterations * sizeof(double));
        sparse_matrix *Pd= PrepareHybrid(nnz, rows,csrCidx, csrRptr,csrVal);
        for (int i = 0; i < iterations; i++) {
          time_coo[i] = test_coo(coov, cooi, cooj, nnz, x, rows, d, Fserial);
          test_pq(weight, weight_ptr, weight_ind, x, N, bs, num_blocks, rows, d,
                  Fserial, timeInfo);
          gettimeofday(&t1, NULL);

          gpu_hybrid_spmv<coord>(Pd->elements_in_rows,Pd->coo_size,dx,rows,hybridy,Pd->ell_cols,Pd->ell_data,Pd->coo_data,Pd->coo_row_ids,Pd->coo_col_ids,d);
          cudaDeviceSynchronize();
          gettimeofday(&t2, NULL);
          elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
          elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
          time_hybrid[i]=elapsedTime;
          time_bsr_col[i] = timeInfo[1];
          time_bsr_row[i] = timeInfo[0];
        }

        for (int i = 0; i < iterations; i++) {
          printf("%lf ", time_coo[i]);
        }
        printf("\n");

        for (int i = 0; i < iterations; i++) {
          printf("%lf ", time_bsr_row[i]);
        }
        printf("\n");

        for (int i = 0; i < iterations; i++) {
          printf("%lf ", time_bsr_col[i]);
        }

        printf("\n");
        for (int i = 0; i < iterations; i++) {
          printf("%lf ", time_hybrid[i]);
        }

        printf("\n");


        free(time_coo);
        free(time_bsr_col);
        free(time_bsr_row);
        free(time_hybrid);


        printf("Hybrid elapsedTime=%lf\n", elapsedTime);

        coord* result2=(coord *)malloc(sizeof(coord)*rows*d);
        cudaMemcpy(result2, hybridy, rows*d * sizeof (coord),cudaMemcpyDeviceToHost);

        printf("Hybrid error %f\n",maxerror(result2,Fserial , rows,  d) );

      //------------------------------------------------------------------------------
      free(weight);
  free(weight_ind);
  free(weight_ptr);
  free(x);
  free(coov);
  free(cooi);
  free(cooj);
  free(Fserial);
  free(csrVal);
  free(csrRptr);
  free(csrCidx);
  cudaFree(bsrRowPtrC);
  cudaFree(bsrValC);
  cudaFree(bsrColIndC);


  return 0;
}
