
#include "nuconv.hpp"
#include "gridding.hpp"
#include "non_periodic_conv.hpp"

void nuconvCPU( coord *PhiScat, coord *y, coord *VScat,
             uint32_t *ib, uint32_t *cb,
             int n, int d, int m, int np, int nGridDim){

  // ~~~~~~~~~~ normalize coordinates (inside bins)
  coord maxy = 0;
  for (int i = 0; i < n*d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for(int i=0; i<n*d;i++){
    y[i] /= maxy;

  // ~~~~~~~~~~ scale them from 0 to ng-1

  if (1 == y[i])
    y[i] = y[i] - std::numeric_limits<coord>::epsilon();

  y[i] *= (nGridDim-1);
  }
  for (int i = 0; i< n*d; i++)
    if ( (y[i] >= nGridDim-1) || (y[i] < 0) ) exit(1);

  // ~~~~~~~~~~ find exact h

  double h = maxy / (nGridDim - 1 - std::numeric_limits<coord>::epsilon() );


  // ~~~~~~~~~~ scat2grid
  int szV = pow( nGridDim+2, d ) * m;
  coord *VGrid = static_cast<coord *> ( calloc( szV * np, sizeof(coord) ) );



  switch (d) {

  case 1:
    if (nGridDim <= GRID_SIZE_THRESHOLD){
      s2g1dCpu( VGrid, y, VScat, nGridDim+2, np, n, d, m );
      }
    else
      printf("papa\n" );
      //s2g1drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 2:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
      printf("papa\n" );

      //s2g2d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
      printf("papa\n" );

      //s2g2drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  case 3:
    if (nGridDim <= GRID_SIZE_THRESHOLD)
        printf("papa\n" );

      //s2g3d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    else
        printf("papa\n" );
      //s2g3drb( VGrid, y, VScat, ib, cb, nGridDim+2, np, n, d, m );
    break;

  }

  // ---- reduction across processors
  for( int i=0; i < szV ; i++ )
    for (int j=1; j<np; j++)
      VGrid[i] += VGrid[ j*szV + i ];

  VGrid = static_cast<coord *> ( realloc( VGrid, szV*sizeof(coord) ) );


  // ~~~~~~~~~~ grid2grid
  coord *PhiGrid = static_cast<coord *> ( calloc( szV, sizeof(coord) ) );
  uint32_t * const nGridDims = new uint32_t [d]();
  for(int i=0;i<d;i++){
    nGridDims[i] = nGridDim + 2;
  }


  switch (d) {

  case 1:
    conv1dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
    break;

  case 2:
  printf("papa\n" );

    //conv2dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
    break;

  case 3:
  printf("papa\n" );

    //conv3dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
    break;

  }



  // ~~~~~~~~~~ grid2scat

  switch (d) {

  case 1:
    g2s1dCpu( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;

  case 2:
    printf("papa\n" );
    //g2s2d( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;

  case 3:
    printf("papa\n" );
    //g2s3d( PhiScat, PhiGrid, y, nGridDim+2, n, d, m );
    break;

  }


  // ~~~~~~~~~~ deallocate memory
  free( VGrid );
  free( PhiGrid );

  delete nGridDims;

}
