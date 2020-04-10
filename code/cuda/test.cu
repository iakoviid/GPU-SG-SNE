#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#define LIMIT_SEQ 512

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
    y[i] = ((double) rand() / (RAND_MAX))*100;

  return y;
}
 __global__ void generateBoxIdx( uint64_t* Code,const double * Y,double scale,const int nPts,const int nDim,const int nGrid,const double multQuant,const uint32_t qLevel)
{
  register uint64_t C[3];
  register double Yscale;
  for(register int TID = threadIdx.x + blockIdx.x * blockDim.x; TID < nPts ;TID+=gridDim.x*blockDim.x){
    for(int j=0;j<nDim;j++){
      Yscale=Y[TID+j*nPts]/scale;
      if(Yscale>=1){ Yscale=1 - 0.00000000000001;
        //printf("Yscale= %lf\n",Yscale );

    }
      C[j]=(uint64_t) abs( floor( multQuant * Yscale ) );
    }
    switch (nDim) {

    case 1:
      Code[TID]= (uint64_t) C[0];

    case 2:
      Code[TID]= ( ( (uint64_t) C[1] ) << qLevel ) |( ( (uint64_t) C[0] )           );

    case 3:
      Code[TID]= ( ( (uint64_t) C[2] ) << 2*qLevel ) |
                ( ( (uint64_t) C[1] ) <<   qLevel ) |
                ( (uint64_t) C[0] )             ;


  }}
  return ;

}

uint64_t tangleCode( const double  * const YScat,
                     const double  scale,
                     const double  multQuant,
                     const uint32_t nGrid,
                     const uint32_t nDim)
                      {

  uint32_t qLevel = ceil(log(nGrid)/log(2));

  uint64_t C[3];

  for ( uint32_t j=0; j<nDim; j++){

    // get scaled input
    double Yscale = YScat[j] / scale;
    if (Yscale >= 1) {Yscale = 1 - std::numeric_limits<double>::epsilon();
        //printf("Yscale= %lf\n",Yscale );
    }

    // scale data points
    C[j] = (uint32_t) abs( floor( multQuant * Yscale ) );
  }

  switch (nDim) {

  case 1:
    return (uint64_t) C[0];

  case 2:
    return ( ( (uint64_t) C[1] ) << qLevel ) |
           ( ( (uint64_t) C[0] )           );

  case 3:
    return ( ( (uint64_t) C[2] ) << 2*qLevel ) |
           ( ( (uint64_t) C[1] ) <<   qLevel ) |
           ( ( (uint64_t) C[0] )             );

  default:
    return 0;
  }

}
void quantizeAndComputeCodes( uint64_t * const C,
                              const double * const YScat,
                              const double scale,
                              const uint32_t nPts,
                              const uint32_t nDim,
                              const uint32_t nGrid )
                               {

  // get quantization multiplier
  double multQuant = nGrid - 1 - std::numeric_limits<double>::epsilon();

  // add codes and ID to struct to sort them
  for(int i=0; i<nPts; i++){
    C[i] = tangleCode( &YScat[i*nDim], scale, multQuant, nGrid, nDim );
  }
}

template<typename dataval>
void doSort( uint64_t * const Cs, uint64_t * const Ct,
             uint32_t * const Ps, uint32_t * const Pt,
             dataval   * const Ys, dataval   * const Yt,
             uint32_t prev_off,
             const uint32_t nbits, const uint32_t sft,
             const uint32_t n, const uint32_t d,
             uint32_t nb )
             {

  // prepare bins
  uint32_t nBin = (0x01 << (nbits));
  // uint32_t *BinCursor  = new uint32_t[ nBin ]();
  uint32_t *BinCursor = (uint32_t *)calloc( nBin, sizeof(uint32_t) );

  // current code
  uint32_t *code = new uint32_t[d]();

  // get mask for required number of bits
  uint64_t mask = ( 0x01 << (nbits) ) - 1;

  for(int i=0; i<n; i++) {
    uint32_t const ii = (Cs[i] >> sft) & mask;
    BinCursor[ii]++;
  }

  // scan prefix (can be better!)
  int offset = 0;
  for(int i=0; i<nBin; i++) {
    int const ss = BinCursor[i];
    BinCursor[i] = offset;
    offset += ss;
  }

  // permute points
  for(int i=0; i<n; i++){
    uint32_t const ii = (Cs[i] >> sft) & mask;
    Ct[BinCursor[ii]] = Cs[i];
    for(int kapa=0;kapa<d;kapa++){
    Yt[BinCursor[ii]*d+kapa] = Ys[i*d+kapa];}
    Pt[BinCursor[ii]] = Ps[i];
    BinCursor[ii]++;
  }

  if (sft>=nbits){

    offset = 0;
    for(int i=0; i<nBin; i++){
      uint32_t nPts = BinCursor[i] - offset;

      if ( nPts > LIMIT_SEQ ){
         doSort( &Ct[offset], &Cs[offset],
                           &Pt[offset], &Ps[offset],
                           &Yt[offset*d], &Ys[offset*d],
                           prev_off + offset,
                           nbits, sft-nbits, nPts, d, nb );
      } else if ( nPts > 0 ){
        doSort( &Ct[offset], &Cs[offset],
                &Pt[offset], &Ps[offset],
                &Yt[offset*d], &Ys[offset*d],
                prev_off + offset,
                nbits, sft-nbits, nPts, d, nb );


      }
      offset = BinCursor[i];
    }
  }

  ;

  // delete BinCursor;
  free( BinCursor );
  delete code;

}

template<typename dataval>
void doSort_top( uint64_t * const Cs, uint64_t * const Ct,
                 uint32_t * const Ps, uint32_t * const Pt,
                 dataval   * const Ys, dataval   * const Yt,
                 uint32_t prev_off,
                 const uint32_t nbits, const uint32_t sft,
                 const uint32_t n, const uint32_t d,
                 uint32_t nb, uint32_t np )
                 {

  // prepare bins
  uint32_t nBin = (0x01 << (nbits));

  // retrive active block per thread
  int m = (int) std::ceil ( (float) n / (float)np );

  uint32_t *BinCursor = (uint32_t *)calloc( nBin*np, sizeof(uint32_t) );

  // current code
  uint32_t *code = new uint32_t[d]();

  // get mask for required number of bits
  uint64_t mask = ( 0x01 << (nbits) ) - 1;

  for (int i=0; i<np; i++){
    int size = ((i+1)*m < n) ? m : (n - i*m);
    for(int j=0; j<size; j++) {
      uint32_t const ii = ( Cs[ i*m + j ] >> sft ) & mask;
      BinCursor[ i*nBin + ii ]++;
    }
  }

  int offset = 0;
  for (int i=0; i<nBin; i++){
    for(int j=0; j<np; j++) {
      int const ss = BinCursor[j*nBin + i];
      BinCursor[j*nBin + i] = offset;
      offset += ss;
    }
  }

  // permute points
  for (int j=0; j<np; j++){
    int size = ((j+1)*m < n) ? m : (n - j*m);
    for(int i=0; i<size; i++){
      uint32_t const idx = j*m + i;
      uint32_t const ii = (Cs[idx] >> sft) & mask;
      uint32_t const jj = BinCursor[j*nBin + ii];
      Ct[jj] = Cs[idx];
      for(int kapa=0;kapa<d;kapa++){
        Yt[jj*d+kapa] = Ys[idx*d+kapa];
      }
      Pt[jj] = Ps[idx];
      BinCursor[j*nBin + ii]++;
    }
  }

  if (sft>=nbits){

    offset = 0;
    for(int i=0; i<nBin; i++){
      uint32_t nPts = BinCursor[(np-1)*nBin + i] - offset;

      if ( nPts > LIMIT_SEQ ){
         doSort( &Ct[offset], &Cs[offset],
                           &Pt[offset], &Ps[offset],
                           &Yt[offset*d], &Ys[offset*d],
                           prev_off + offset,
                           nbits, sft-nbits, nPts, d, nb );
      } else if ( nPts > 0 ){
        doSort( &Ct[offset], &Cs[offset],
                &Pt[offset], &Ps[offset],
                &Yt[offset*d], &Ys[offset*d],
                prev_off + offset,
                nbits, sft-nbits, nPts, d, nb );


      }
      offset = BinCursor[(np-1)*nBin + i];
    }
  }


  // delete BinCursor;
  free( BinCursor );
  delete code;

}
int main(int argc, char **argv){
  int d=atoi(argv[1]);
  int N=1<<atoi(argv[2]);
  double * y=generateRandomCoord(N,d);
  y[1]=100;
  y[2]=100;
  y[3]=100;
  double* yc=(double *)malloc(N*d*sizeof(double));
  for(int i=0;i<N;i++){
    for(int j=0;j<d;j++){
      yc[i+N*j]=y[i*d+j];
      printf("%lf  ",y[i*d+j] );
    }
    printf("\n" );
  }
  double* y_d;
  int n=N;
  CUDA_CALL(cudaMallocManaged(&y_d,d*n*sizeof(double)));
  CUDA_CALL(cudaMemcpy(y_d,yc,n*d*sizeof(double), cudaMemcpyHostToDevice));
  uint64_t *Codes;
  CUDA_CALL(cudaMallocManaged(&Codes,n*sizeof(uint64_t)));
  int nGrid=10;
  double multQuant = nGrid - 1 - std::numeric_limits<double>::epsilon();
  int threads=1<<atoi(argv[3]);
  int blocks=1<<atoi(argv[4]);
  uint32_t qLevel=0;
  qLevel = ceil(log(nGrid)/log(2));
  generateBoxIdx<<<blocks, threads>>>(Codes,y_d,100,N,d,nGrid,multQuant,qLevel);
  uint64_t *Codes1;
  Codes1=(uint64_t *)malloc(sizeof(uint64_t)* n);
  CUDA_CALL(cudaMemcpy(Codes1,Codes,sizeof(uint64_t)* n, cudaMemcpyDeviceToHost));

  uint64_t *Codes2;
  Codes2=(uint64_t *)malloc(sizeof(uint64_t)* n);
  quantizeAndComputeCodes(Codes2,y,100,n,d,nGrid);
  printf("----------------------------------------\n" );
  for(int i=0;i<n;i++){
    if(Codes1[i]!=Codes2[i]){
      printf("------------Error i=%d----------\n",i );}
  }

  thrust::device_ptr<double> yVec_ptr(y_d);
  thrust::device_vector<double> yVec_d(yVec_ptr,yVec_ptr+n*d);
  thrust::device_vector<double>::iterator iter=thrust::max_element(yVec_d.begin(),yVec_d.end());
  unsigned int position= iter-yVec_d.begin();
  double maxy=yVec_d[position];
  double* miny=(double *)malloc(sizeof(double)*d);
  for(int j=0;j<d;j++){

    thrust::device_vector<double>::iterator iter=thrust::min_element(yVec_d.begin()+j*n,yVec_d.begin()+n*(j+1));

    position=iter-(yVec_d.begin());

    miny[j]=yVec_d[position];
    printf("%lf\n",miny[j] );
  }
  cudaDeviceSynchronize();

  thrust ::device_ptr<uint64_t> Codes_ptr(Codes);
  thrust::device_vector<uint32_t> iPerm(n);
  thrust::sequence(iPerm.begin(), iPerm.end());

  switch (d) {

  case 1:
    thrust ::stable_sort_by_key(Codes_ptr, Codes_ptr + n, make_zip_iterator(make_tuple(yVec_ptr,iPerm.begin())));

  case 2:
    thrust ::stable_sort_by_key(Codes_ptr, Codes_ptr + n, make_zip_iterator(make_tuple(yVec_ptr, yVec_ptr+n,iPerm.begin())));

  case 3:
    thrust ::stable_sort_by_key(Codes_ptr, Codes_ptr + n, make_zip_iterator(make_tuple(yVec_ptr, yVec_ptr+n,yVec_ptr+n,iPerm.begin())));


}
  CUDA_CALL(cudaMemcpy(Codes1,Codes,sizeof(uint64_t)* n, cudaMemcpyDeviceToHost));
  CUDA_CALL(cudaMemcpy(yc,y_d,sizeof(double)*d* n, cudaMemcpyDeviceToHost));
  uint32_t* perm=(uint32_t *)malloc(n*sizeof(uint32_t));
  CUDA_CALL(cudaMemcpy(perm,thrust::raw_pointer_cast(iPerm.data()),sizeof(uint32_t)* n, cudaMemcpyDeviceToHost));

  for(int i=0;i<N;i++){
    printf("%ld ",Codes1[i] );
    for(int j=0;j<d;j++){

      printf("%lf  ",yc[i+N*j] );
    }
    printf(" --> %d ",perm[i] );
    printf("\n" );
  }
  uint64_t  * const C2     = (uint64_t *  ) malloc(sizeof(uint64_t) * 1    * n);
  double   * const Y2     = (double   * ) malloc(sizeof(double)  * d* n);
  uint32_t  * const iPerm2 = (uint32_t  * ) malloc(sizeof(uint32_t) * 1    * n);
  uint32_t  * const iPermpa = (uint32_t  * ) malloc(sizeof(uint32_t) * 1    * n);
  for(int i=0;i<n;i++){
    iPermpa[i]=i;
  }
  doSort_top( Codes2, C2, iPermpa, iPerm2, y, Y2, 0,
              qLevel, (d-1)*qLevel, n, d, nGrid, 1 );

  printf("----------------------------------------\n" );

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

  uint32_t *ib;// Starting index of box (along last dimension)
  uint32_t *cb;//Number of scattered points per box (along last dimension)



return 0;

}
