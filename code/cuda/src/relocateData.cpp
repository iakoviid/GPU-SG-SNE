/*!
  \file   dataReloc.cpp
  \brief  Fast data relocation modules.

  \author Dimitris Floros
  \date   2019-06-24
*/


#include "relocateData.hpp"
#include <sys/time.h>

#define LIMIT_SEQ 512

// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ ALLOCATION UTILITIES

void* parallel_malloc(size_t items, size_t size, const char * message){
  void *ptr = malloc(items*size);
  if(ptr == 0){
    printf("Out of memory at %s\n", message);
  }
  return ptr;
}

void* parallel_calloc(size_t items, size_t size, const char * message){
  void *ptr = calloc(items,size);
  if(ptr == 0){
    printf("Out of memory at %s\n", message);
  }
  return ptr;
}

void parallel_free(void *ptr){
  free(ptr);
}

// %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOCAL FUNCTIONS (NOT IN HEADERS)

template<typename dataval>
uint64_t tangleCode( const dataval  * const YScat,
                     const dataval  scale,
                     const dataval  multQuant,
                     const uint32_t nGrid,
                     const uint32_t nDim)
                      {

  uint32_t qLevel = ceil(log(nGrid)/log(2));

  uint64_t C[3];

  for ( uint32_t j=0; j<nDim; j++){

    // get scaled input
    dataval Yscale = YScat[j] / scale;
    if (Yscale >= 1) Yscale = 1 - std::numeric_limits<dataval>::epsilon();

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


template<typename dataval>
void quantizeAndComputeCodes( uint64_t * const C,
                              const dataval * const YScat,
                              const dataval scale,
                              const uint32_t nPts,
                              const uint32_t nDim,
                              const uint32_t nGrid )
                               {

  // get quantization multiplier
  dataval multQuant = nGrid - 1 - std::numeric_limits<dataval>::epsilon();

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
  uint32_t *BinCursor = (uint32_t *)parallel_calloc( nBin, sizeof(uint32_t), "Bin memory" );

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
    for(int j=0;j<d;j++){
      Yt[BinCursor[ii]*d+j] = Ys[i*d +j];
    }
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


  // delete BinCursor;
  parallel_free( BinCursor );
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

  uint32_t *BinCursor = (uint32_t *)parallel_calloc( nBin*np, sizeof(uint32_t), "Bin memory" );

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
  for(int i=0;i<nBin;i++){
    //printf("BinCursor[%d]=%d\n",i,BinCursor[i] );
  }
  //printf("\n" );

  int offset = 0;
  for (int i=0; i<nBin; i++){
    for(int j=0; j<np; j++) {
      int const ss = BinCursor[j*nBin + i];
      BinCursor[j*nBin + i] = offset;
      offset += ss;
    }
  }
  for(int i=0;i<nBin;i++){
    //printf("BinCursor[%d]=%d\n",i,BinCursor[i] );
  }
  //printf("\n" );

  // permute points
  for (int j=0; j<np; j++){
    int size = ((j+1)*m < n) ? m : (n - j*m);
    for(int i=0; i<size; i++){
      uint32_t const idx = j*m + i;
      uint32_t const ii = (Cs[idx] >> sft) & mask;
      uint32_t const jj = BinCursor[j*nBin + ii];
      Ct[jj] = Cs[idx];
      for(int pa=0;pa<d;pa++){
        Yt[jj*d +pa] = Ys[idx*d+pa];

      }
      Pt[jj] = Ps[idx];
      BinCursor[j*nBin + ii]++;
    }
  }

  //printf("\n" );
  //return;
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
  parallel_free( BinCursor );
  delete code;

}


__inline__
uint32_t untangleLastDim( const uint64_t C,
                          const uint32_t nDim,
                          const uint32_t qLevel )
                          {

  uint32_t Cout = 0;

  switch (nDim) {

  case 1:
    Cout = (uint32_t) C;
    break;

  case 2:
    {
      uint64_t mask = (1<<2*qLevel) - 1;

      Cout = (uint32_t) ( ( C & mask ) >> qLevel );
      break;
    }

  case 3:
    {
      uint64_t mask = (1<<3*qLevel) - 1;

      Cout = (uint32_t) ( ( C & mask ) >> 2*qLevel );
      break;
    }

  default:
    {
      std::cerr << "Supporting up to 3D" << std::endl;
      exit(1);
    }

  }

  return Cout;

}


void gridSizeAndIdx( uint32_t * const ib,
                     uint32_t * const cb,
                     uint64_t const * const C,
                     const uint32_t nPts,
                     const uint32_t nDim,
                     const uint32_t nGridDim )
                     {

  uint32_t qLevel = ceil(log(nGridDim)/log(2));
  uint32_t idxCur = -1;
 uint32_t a=-1;
  for (uint32_t i = 0; i < nPts; i++){

    uint32_t idxNew = untangleLastDim( C[i], nDim, qLevel );

    cb[idxNew]++;
    if(idxNew!=a){
      //printf("idxNew=%d i=%d \n",idxNew,i );
      a=idxNew;
    }

    if (idxNew != idxCur) ib[idxNew+1] = i+1;

  }

}


//! Coarse-grid quantization and data relocation
/*!
*/
void relocateCoarseGridCPU( coord  ** Yptr,        // Scattered point coordinates
                         uint32_t ** iPermptr,    // Data relocation permutation
                         uint32_t *ib,            // Starting index of box (along last dimension)
                         uint32_t *cb,            // Number of scattered points per box (along last dimension)
                         int nPts,        // Number of data points
                         int nGridDim,    // Grid dimensions (+1)
                         int nDim,      // Number of dimensions
                         int np) {      // Number of processors

  coord  * Y = *Yptr;
  uint32_t * iPerm = *iPermptr;

  uint64_t  * const C1     = (uint64_t *  ) malloc(sizeof(uint64_t) * 1    * nPts);
  uint64_t  * const C2     = (uint64_t *  ) malloc(sizeof(uint64_t) * 1    * nPts);

  coord   * const Y2     = (coord   * ) malloc(sizeof(coord)  * nDim * nPts);
  uint32_t  * const iPerm2 = (uint32_t  * ) malloc(sizeof(uint32_t) * 1    * nPts);;



  // ========== get scaling factor
  coord scale = std::numeric_limits<coord>::min();

  for (int i=0; i<nPts; i++)
    for (int j=0; j<nDim; j++)
      if (scale < Y[i*nDim+j]) scale = Y[i*nDim+j];

  // ========== quantize data and compute codes
  quantizeAndComputeCodes( C1, Y, scale, nPts, nDim, nGridDim );

  // ========== perform binning and relocation
  uint32_t qLevel = ceil(log(nGridDim)/log(2));
  struct timeval t1, t2;
  double elapsedTime;
  gettimeofday(&t1, NULL);
  doSort_top(C1, C2, iPerm, iPerm2, Y, Y2, 0, qLevel, (nDim - 1) * qLevel, nPts,
             nDim, nGridDim, 1);
  gettimeofday(&t2, NULL);

  elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
  //printf("SORT time milliseconds %f\n", elapsedTime);
  // ========== deallocate memory and return correct pointers

  if ( (nDim%2) == 1 ){

    // ========== get starting index and size of each grid box
    gridSizeAndIdx( ib, cb, C2, nPts, nDim, nGridDim );

    free( Y     );
    free( iPerm );

    *Yptr = Y2;
    *iPermptr = iPerm2;

  } else {

    // ========== get starting index and size of each grid box
    gridSizeAndIdx( ib, cb, C1, nPts, nDim, nGridDim );

    free( Y2 );
    free(iPerm2);
  }

  //printf("Suck\n" );

  free( C1 ); free( C2 );
  //printf("ooa\n" );
  //return;

}
