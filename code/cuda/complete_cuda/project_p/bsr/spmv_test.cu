//////////////////////////////////////////////////////////////////////////
// In this file we compute Gemv sparse matrix with cuda for two kernels:
// CSR and BCSR
// We use a CPU COO version to test the accuracy of the code.
//////////////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////////////
// COO part
//////////////////////////////////////////////////////////////////////////
#include <sys/time.h>

#include <cublas_v2.h>
#include <cuda.h>
#include <cusparse_v2.h>
#include <stdio.h>
#include <stdlib.h>
#include <cassert>
#include <cstdio>
#include <cstring> // memset
#include "template.cuh"
#include "mmio.h"
#define bs 8

#define Min(x, y) ((x) < (y) ? (x) : (y))
#define Max(x, y) ((x) > (y) ? (x) : (y))
#define Abs(x) ((x) > (0) ? (x) : -(x))


#define FULL_WARP_MASK 0xFFFFFFFF

template <class T>
__device__ T warp_reduce (T val)
{
  /**
   *  For a thread at lane X in the warp, __shfl_down_sync(FULL_MASK, val, offset) gets
   *  the value of the val variable from the thread at lane X+offset of the same warp.
   *  The data exchange is performed between registers, and more efficient than going
   *  through shared memory, which requires a load, a store and an extra register to
   *  hold the address.
   */
  for (int offset = warpSize / 2; offset > 0; offset /= 2)
    val += __shfl_down_sync (FULL_WARP_MASK, val, offset);

  return val;
}

template <typename coord, typename matidx>
__global__ void bsr1 (
  matidx n_block_rows,
  const matidx * __restrict__ col_ids,
  const matidx * __restrict__ row_ptr,
  const coord * __restrict__ data,
  const coord * __restrict__ x,
  coord *y)
{
  printf("TATA\n" );

  const matidx idx = blockIdx.x * blockDim.x + threadIdx.x;
  const matidx row = (idx / 32) % bs;
  const matidx lane = idx % 32;
  const matidx block_row = (idx / 32) / bs;
  printf("threadIdx =%d block_row=%d \n",threadIdx.x,block_row );
  const matidx first_block = row_ptr[block_row];
  const matidx last_block = row_ptr[block_row + 1];

  coord local_out = 0.0;

  if (row < bs && block_row < n_block_rows)
    {
      for (matidx loc_col = lane; loc_col < bs * (last_block - first_block); loc_col += 32)
        {
          const matidx block = first_block + loc_col / bs;
          const matidx c = loc_col % bs;
          const matidx col = col_ids[block] * bs + c;
          local_out += x[col] * data[block * bs * bs + row * bs + c];
        }
    }

  local_out = warp_reduce (local_out);

  if (row < bs && block_row < n_block_rows && lane == 0)
    y[block_row * bs + row] = local_out;

}


////////////////////// CUDA ERROR /////////////////////////////////////////

static void CudaCheckCore(cudaError_t code, const char *file, int line) {
  if (code != cudaSuccess) {
    fprintf(stderr, "Cuda Error %d : %s %s %d\n", code,
            cudaGetErrorString(code), file, line);
    exit(code);
  }
}

#define CudaCheck(test)                                                        \
  { CudaCheckCore((test), __FILE__, __LINE__); }
#define CudaCheckAfterCall()                                                   \
  { CudaCheckCore((cudaGetLastError()), __FILE__, __LINE__); }

////////////////////// CUDA SPARSE ERROR ///////////////////////////////////
/*
static const char * cusparseGetErrorString(cusparseStatus_t error)
{
    // Read more at:
http://docs.nvidia.com/cuda/cusparse/index.html#ixzz3f79JxRar switch (error)
    {
    case CUSPARSE_STATUS_SUCCESS:
        return "The operation completed successfully.";
    case CUSPARSE_STATUS_NOT_INITIALIZED:
        return "The cuSPARSE library was not initialized. This is usually caused
by the lack of a prior call, an error in the CUDA Runtime API called by the
cuSPARSE routine, or an error in the hardware setup.\n" \ "To correct: call
cusparseCreate() prior to the function call; and check that the hardware, an
appropriate version of the driver, and the cuSPARSE library are correctly
installed.";

    case CUSPARSE_STATUS_ALLOC_FAILED:
        return "Resource allocation failed inside the cuSPARSE library. This is
usually caused by a cudaMalloc() failure.\n"\ "To correct: prior to the function
call, deallocate previously allocated memory as much as possible.";

    case CUSPARSE_STATUS_INVALID_VALUE:
        return "An unsupported value or parameter was passed to the function (a
negative vector size, for example).\n"\ "To correct: ensure that all the
parameters being passed have valid values.";

    case CUSPARSE_STATUS_ARCH_MISMATCH:
        return "The function requires a feature absent from the device
architecture; usually caused by the lack of support for atomic operations or
double precision.\n"\ "To correct: compile and run the application on a device
with appropriate compute capability, which is 1.1 for 32-bit atomic operations
and 1.3 for double precision.";

    case CUSPARSE_STATUS_MAPPING_ERROR:
        return "An access to GPU memory space failed, which is usually caused by
a failure to bind a texture.\n"\ "To correct: prior to the function call, unbind
any previously bound textures.";

    case CUSPARSE_STATUS_EXECUTION_FAILED:
        return "The GPU program failed to execute. This is often caused by a
launch failure of the kernel on the GPU, which can be caused by multiple
reasons.\n"\ "To correct: check that the hardware, an appropriate version of the
driver, and the cuSPARSE library are correctly installed.";

    case CUSPARSE_STATUS_INTERNAL_ERROR:
        return "An internal cuSPARSE operation failed. This error is usually
caused by a cudaMemcpyAsync() failure.\n"\ "To correct: check that the hardware,
an appropriate version of the driver, and the cuSPARSE library are correctly
installed. Also, check that the memory passed as a parameter to the routine is
not being deallocated prior to the routine’s completion.";

    case CUSPARSE_STATUS_MATRIX_TYPE_NOT_SUPPORTED:
        return "The matrix type is not supported by this function. This is
usually caused by passing an invalid matrix descriptor to the function.\n"\ "To
correct: check that the fields in cusparseMatDescr_t descrA were set
correctly.";
    }

    return "<unknown>";
}*/
static void CudaSparseCheckCore(cusparseStatus_t code, const char *file,
                                int line) {
  if (code != CUSPARSE_STATUS_SUCCESS) {
    fprintf(stderr, "Cuda Error %d : %s %s %d\n", code,
            cusparseGetErrorString(code), file, line);
    exit(code);
  }
}

#define CudaSparseCheck(test)                                                  \
  { CudaSparseCheckCore((test), __FILE__, __LINE__); }

////////////// Alloc and copy ////////////////////////////////////////////////

template <class ObjectType>
ObjectType *allocAndCopy(const ObjectType src[], const int size) {
  ObjectType *dest = NULL;
  CudaCheck(cudaMalloc(&dest, size * sizeof(ObjectType)));
  CudaCheck(
      cudaMemcpy(dest, src, size * sizeof(ObjectType), cudaMemcpyHostToDevice));
  return dest;
}

template <class ObjectType> ObjectType *alloc(const int size) {
  ObjectType *dest = NULL;
  CudaCheck(cudaMalloc(&dest, size * sizeof(ObjectType)));
  return dest;
}

template <class ObjectType>
ObjectType *allocAndCopyPart(const ObjectType src[], const int size,
                             const int allocSize) {
  ObjectType *dest = NULL;
  assert(size <= allocSize);
  CudaCheck(cudaMalloc(&dest, allocSize * sizeof(ObjectType)));
  CudaCheck(
      cudaMemcpy(dest, src, size * sizeof(ObjectType), cudaMemcpyHostToDevice));
  CudaCheck(
      cudaMemset(&dest[size], 0, (allocSize - size) * sizeof(ObjectType)));
  return dest;
}
//////////////////////////////////////////////////////////////////////////
// COO part
//////////////////////////////////////////////////////////////////////////

#include <algorithm>

struct Ijv {
  int i, j;
  double v;
};

bool IjvComp(const Ijv &v1, const Ijv &v2) {
  return v1.i < v2.i || (v1.i == v2.i && v1.j < v2.j);
}

struct COOArrays {
  int m;
  int nnz;
  double *val; /*values(NNZ)*/
  int *rowind; /*i(NNZ)*/
  int *colind; /*j(NNZ)*/

  COOArrays() {
    val = NULL;
    rowind = NULL;
    colind = NULL;
  }

  ~COOArrays() {
    delete[] val;
    delete[] rowind;
    delete[] colind;
  }

  void sortToRowMajor() {

    Ijv *ijvs = new Ijv[nnz];
    for (int idxCopy = 0; idxCopy < nnz; ++idxCopy) {
      ijvs[idxCopy].i = rowind[idxCopy];
      ijvs[idxCopy].j = colind[idxCopy];
      ijvs[idxCopy].v = val[idxCopy];
    }

    std::sort(ijvs, ijvs + nnz, IjvComp);

    for (int idxCopy = 0; idxCopy < nnz; ++idxCopy) {
      rowind[idxCopy] = ijvs[idxCopy].i;
      colind[idxCopy] = ijvs[idxCopy].j;
      val[idxCopy] = ijvs[idxCopy].v;
    }

    delete[] ijvs;
  }
};

void compute_COO(COOArrays &coo, double *x, double *y) {
  for (int idxVal = 0; idxVal < coo.nnz; ++idxVal) {
    y[coo.rowind[idxVal]] += x[coo.colind[idxVal]] * coo.val[idxVal];
  }
}

//////////////////////////////////////////////////////////////////////////
// COO part
//////////////////////////////////////////////////////////////////////////

struct CRSArrays {
  int m;              //< the dim of the matrix
  int nnz;            //< the number of nnz (== ia[m])
  double *cu_csrValA; //< the values (of size NNZ)
  int *cu_csrRowPtrA; //< the usual rowptr (of size m+1)
  int *cu_csrColIndA; //< the colidx of each NNZ (of size nnz)

  cudaStream_t streamId;
  cusparseHandle_t cusparseHandle;

  CRSArrays() {
    cu_csrValA = NULL;
    cu_csrRowPtrA = NULL;
    cu_csrColIndA = NULL;

    // Create sparse handle (needed to call sparse functions
    streamId = 0;
    cusparseHandle = 0;
    CudaSparseCheck(cusparseCreate(&cusparseHandle));
    CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
  }

  ~CRSArrays() {
    CudaCheck(cudaFree(cu_csrValA));
    CudaCheck(cudaFree(cu_csrRowPtrA));
    CudaCheck(cudaFree(cu_csrColIndA));

    // Destroy sparse handle
    CudaSparseCheck(cusparseDestroy(cusparseHandle));
  }
};

void COO_to_CRS(COOArrays &coo, CRSArrays *crs) {
  // We need COO to be sorted by row (and column)
  coo.sortToRowMajor();

  crs->m = coo.m;
  crs->nnz = coo.nnz;

  // Convert COO to CSR (it is just for the rows idx)
  crs->cu_csrRowPtrA = alloc<int>(coo.m + 1);
  {
    int *cu_cooRowIndA = allocAndCopy(coo.rowind, coo.nnz);
    CudaSparseCheck(cusparseXcoo2csr(crs->cusparseHandle, cu_cooRowIndA,
                                     coo.nnz, coo.m, crs->cu_csrRowPtrA,
                                     CUSPARSE_INDEX_BASE_ZERO));
    CudaCheck(cudaFree(cu_cooRowIndA));
  }
  // Copy cols idx and values that are unchanged
  crs->cu_csrValA = allocAndCopy(coo.val, coo.nnz);
  crs->cu_csrColIndA = allocAndCopy(coo.colind, coo.nnz);
}

double compute_CRS(CRSArrays &crs, double *x, double *y) {
  // For blas 2 gemv y = alpha.x.A + Beta.y
  const double alpha = 1.0;
  const double beta = 0.0;
  // Copy input
  double *cu_x = allocAndCopy(x, crs.m);
  double *cu_y = allocAndCopy(y, crs.m);
  // Init matrix properties
  cusparseMatDescr_t descr = 0;
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv
  float gemvComputeTume = 0;
  {
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, crs.streamId);

    CudaSparseCheck(cusparseDcsrmv(
        crs.cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE, crs.m, crs.m,
        crs.nnz, &alpha, descr, crs.cu_csrValA, crs.cu_csrRowPtrA,
        crs.cu_csrColIndA, cu_x, &beta, cu_y));

    cudaEventRecord(stopTime, crs.streamId);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&gemvComputeTume, startTime, stopTime);
    gemvComputeTume /= 1000.0;
  }
  // Get back result
  CudaCheck(
      cudaMemcpy(y, cu_y, crs.m * sizeof(double), cudaMemcpyDeviceToHost));
  // Dealloc vectors
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));

  return gemvComputeTume;
}

//////////////////////////////////////////////////////////////////////////
// BCSR part
//////////////////////////////////////////////////////////////////////////

struct BCRSArrays {
  int m;
  int nnz;
  int nbBlocks;
  int nbBlockRow;
  int blockSize;

  int *cu_bsrRowPtrC;
  int *cu_bsrColIndC;
  double *cu_bsrValC;

  cudaStream_t streamId;
  cusparseHandle_t cusparseHandle;

  BCRSArrays() {
    cu_bsrRowPtrC = NULL;
    cu_bsrColIndC = NULL;
    cu_bsrValC = NULL;

    // Create sparse handle (needed to call sparse functions
    streamId = 0;
    cusparseHandle = 0;
    CudaSparseCheck(cusparseCreate(&cusparseHandle));
    CudaSparseCheck(cusparseSetStream(cusparseHandle, streamId));
  }

  ~BCRSArrays() {
    CudaCheck(cudaFree(cu_bsrRowPtrC));
    CudaCheck(cudaFree(cu_bsrColIndC));
    CudaCheck(cudaFree(cu_bsrValC));

    // Destroy sparse handle
    CudaSparseCheck(cusparseDestroy(cusparseHandle));
  }
};

void CRS_to_BCRS(CRSArrays &csr, BCRSArrays *bcrs, const int blockSize) {
  bcrs->m = csr.m;
  bcrs->nnz = csr.nnz;
  bcrs->blockSize = blockSize;

  bcrs->nbBlockRow = (csr.m + blockSize - 1) / blockSize;

  cudaMalloc((void **)&bcrs->cu_bsrRowPtrC,
             sizeof(int) * (bcrs->nbBlockRow + 1));

  cusparseMatDescr_t descr = 0;
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);

  int nbNnzBlocks;
  cusparseXcsr2bsrNnz(bcrs->cusparseHandle, CUSPARSE_DIRECTION_COLUMN, csr.m,
                      csr.m, descr, csr.cu_csrRowPtrA, csr.cu_csrColIndA,
                      blockSize, descr, bcrs->cu_bsrRowPtrC, &nbNnzBlocks);
  {
    int firstBlockIdx, lastBlockIdx;
    cudaMemcpy(&lastBlockIdx, bcrs->cu_bsrRowPtrC + bcrs->nbBlockRow,
               sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&firstBlockIdx, bcrs->cu_bsrRowPtrC, sizeof(int),
               cudaMemcpyDeviceToHost);
    assert(firstBlockIdx == 0); // we are in base 0
    assert(nbNnzBlocks == lastBlockIdx - firstBlockIdx);
  }
  bcrs->nbBlocks = nbNnzBlocks;

  CudaCheck(
      cudaMalloc((void **)&bcrs->cu_bsrColIndC, sizeof(int) * nbNnzBlocks));
  CudaCheck(cudaMalloc((void **)&bcrs->cu_bsrValC,
                       sizeof(double) * (blockSize * blockSize) * nbNnzBlocks));
  cusparseDcsr2bsr(bcrs->cusparseHandle, CUSPARSE_DIRECTION_COLUMN, csr.m,
                   csr.m, descr, csr.cu_csrValA, csr.cu_csrRowPtrA,
                   csr.cu_csrColIndA, blockSize, descr, bcrs->cu_bsrValC,
                   bcrs->cu_bsrRowPtrC, bcrs->cu_bsrColIndC);
}

double compute_BSR(BCRSArrays &bcsr, double *x, double *y) {
  // For blas 2 gemv y = alpha.x.A + Beta.y
  const double alpha = 1.0;
  const double beta = 0.0;
  // Copy input
  const int sizeMultipleBlockSize =
      ((bcsr.m + bcsr.blockSize - 1) / bcsr.blockSize) * bcsr.blockSize;
  double *cu_x = allocAndCopyPart(x, bcsr.m, sizeMultipleBlockSize);
  double *cu_y = allocAndCopyPart(y, bcsr.m, sizeMultipleBlockSize);
  // Init matrix properties
  cusparseMatDescr_t descr = 0;
  CudaSparseCheck(cusparseCreateMatDescr(&descr));
  cusparseSetMatType(descr, CUSPARSE_MATRIX_TYPE_GENERAL);
  cusparseSetMatIndexBase(descr, CUSPARSE_INDEX_BASE_ZERO);
  // Compute gemv
  float gemvComputeTume = 0;
  {
    cudaEvent_t startTime, stopTime;
    cudaEventCreate(&startTime);
    cudaEventCreate(&stopTime);
    cudaEventRecord(startTime, bcsr.streamId);

    cusparseDbsrmv(bcsr.cusparseHandle, CUSPARSE_DIRECTION_COLUMN,
                   CUSPARSE_OPERATION_NON_TRANSPOSE, bcsr.nbBlockRow, bcsr.m,
                   bcsr.nbBlocks, &alpha, descr, bcsr.cu_bsrValC,
                   bcsr.cu_bsrRowPtrC, bcsr.cu_bsrColIndC, bcsr.blockSize, cu_x,
                   &beta, cu_y);

    cudaEventRecord(stopTime, bcsr.streamId);
    cudaEventSynchronize(stopTime);
    cudaEventElapsedTime(&gemvComputeTume, startTime, stopTime);
    gemvComputeTume /= 1000.0;
  }
  // Get back result
  CudaCheck(
      cudaMemcpy(y, cu_y, bcsr.m * sizeof(double), cudaMemcpyDeviceToHost));
  // Dealloc vectors
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));

  return gemvComputeTume;
}

/** Simply return the maximum relative diff */
double ChechAccuracy(const double y1[], const double y2[], const int size) {
  double maxDiff = 0;
  for (int idx = 0; idx < size; ++idx) {
    if (y1[idx] != 0.0) {
      maxDiff = Max(maxDiff, Abs((y1[idx] - y2[idx]) / y1[idx]));
    }
  }
  return maxDiff;
}

double compute_BSRkernel(BCRSArrays &bcsr, double *x, double *y) {
  const int sizeMultipleBlockSize =
      ((bcsr.m + bcsr.blockSize - 1) / bcsr.blockSize) * bcsr.blockSize;

  double *cu_x;
  double *cu_y;
  CudaCheck(cudaMalloc(&cu_x, bcsr.m * sizeof(double)));
  CudaCheck(cudaMemcpy(cu_x, x, bcsr.m * sizeof(double), cudaMemcpyHostToDevice));
  CudaCheck(cudaMalloc(&cu_y, bcsr.m * sizeof(double)));
  CudaCheck(cudaMemcpy(cu_y, y, bcsr.m * sizeof(double), cudaMemcpyHostToDevice));
  dim3 block_size = 32;
  dim3 grid_size{};

  grid_size.x = (bcsr.m/8 * 32 + block_size.x - 1) / block_size.x;
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  //bcsr_spmv_kernel_column_by_column_template<<<grid_size, block_size, block_size.x * sizeof(double)>>>(bcsr.cu_bsrColIndC,bcsr.cu_bsrRowPtrC,bcsr.cu_bsrValC,cu_x,cu_y);
  bsr1<<<grid_size, block_size>>> (bcsr.nbBlockRow,bcsr.cu_bsrColIndC,bcsr.cu_bsrRowPtrC,bcsr.cu_bsrValC,cu_x,cu_y);
  cudaDeviceSynchronize();
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms

  // Get back result
  //
  CudaCheck(
      cudaMemcpy(y, cu_y, bcsr.m * sizeof(double), cudaMemcpyDeviceToHost));
  CudaCheck(cudaFree(cu_x));
  CudaCheck(cudaFree(cu_y));
  for(int i =0 ;i<10; i++){
    printf(" %lf\n",y[i] );
  }
  //
  return elapsedTime;

}

/** Compute with COO and call cuda CSR and BCSR */
void computeWithAll(COOArrays &coo, double times[4], double errors[4],
                    const int loop = 1, const int blockSize = 8) {
  CRSArrays crs;
  COO_to_CRS(coo, &crs);

  BCRSArrays bcsr;
  CRS_to_BCRS(crs, &bcsr, blockSize);
  const int dimMulitpleBlock =
      ((coo.m + blockSize - 1) / blockSize) * blockSize;

  double *x = new double[dimMulitpleBlock];
  for (int idx = 0; idx < dimMulitpleBlock; ++idx) {
    x[idx] = 1.0;
  }
  double *y = new double[dimMulitpleBlock];
  double *ycoo = new double[dimMulitpleBlock];

  {
    memset(ycoo, 0, sizeof(double) * coo.m);
    for (int idxLoop = 0; idxLoop < loop; ++idxLoop) {
      compute_COO(coo, x, ycoo);
    }
    times[0] = 0;
    errors[0] = 0;
  }
  {
    memset(y, 0, sizeof(double) * coo.m);
    times[1] = 0;
    for (int idxLoop = 0; idxLoop < loop; ++idxLoop) {
      times[1] += compute_CRS(crs, x, y);
    }
    errors[1] = ChechAccuracy(ycoo, y, coo.m);
  }
  {
    memset(y, 0, sizeof(double) * coo.m);
    times[2] = 0;
    for (int idxLoop = 0; idxLoop < loop; ++idxLoop) {
      times[2] += compute_BSR(bcsr, x, y);
    }
    errors[2] = ChechAccuracy(ycoo, y, coo.m);
  }

  {
    memset(y, 0, sizeof(double) * coo.m);
    times[3] = 0;
    //times[3] = compute_BSRkernel(bcsr, x, y);

    errors[3] = ChechAccuracy(ycoo, y, coo.m);
  }
  delete[] x;
  delete[] y;
  delete[] ycoo;
}

int main(int argc, char **argv) {
  const int dim = 1 << atoi(argv[1]);

  COOArrays coo;
  coo.m = dim;
  coo.nnz = dim;
  double errors[4] = {0, 0, 0, 0};
  double alltimes[4] = {0, 0, 0, 0};
  //double flops[3] = {0, 0, 0};
  //flops[0] = coo.nnz * 2;
  int nz=coo.nnz;
  coo.val = (double *)malloc(sizeof(double)*nz);
  coo.rowind = (int *)malloc(sizeof(int)*nz);
  coo.colind = (int *)malloc(sizeof(int)*nz);
  /*eye matrix*/

  for (int idxRow = 0; idxRow < dim; ++idxRow) {
    coo.val[idxRow] = 1.0;
    coo.rowind[idxRow] = idxRow;
    coo.colind[idxRow] = idxRow;

}

computeWithAll(coo, alltimes, errors);
printf("Finish\n");
printf("Errors= %lf %lf %lf %lf\n", errors[0], errors[1], errors[2], errors[3]);
printf("alltimes= %lf %lf %lf %lf\n", alltimes[0], alltimes[1], alltimes[2],
       alltimes[3]);

return 0;
}
