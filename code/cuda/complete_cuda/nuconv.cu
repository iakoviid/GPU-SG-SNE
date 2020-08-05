
#include "nuconv.cuh"


void nuconv(coord *PhiScat, coord *y, coord *VScat, uint32_t *ib, uint32_t *cb,
            int n, int d, int m, int nGridDim) {

  // ~~~~~~~~~~ normalize coordinates (inside bins)
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


      switch (d) {

      case 1:
        if (nGridDim <= GRID_SIZE_THRESHOLD){
          s2g1d<<<32, 512>>>(VGrid, y, VScat, nGridDim + 2, n, d, d + 1, maxy);
        }
        else
          printf("1\n" );

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



    // ~~~~~~~~~~ grid2grid
    coord *PhiGrid;
    CUDA_CALL(cudaMallocManaged(&PhiGrid, szV * sizeof(coord)));



      switch (d) {

      case 1:
        conv1dnopadcuda(PhiGrid, VGrid, h, nGridDim+2, m, d);

        break;

      case 2:
        //conv2dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
        printf("papa\n" );

        break;

      case 3:
        //conv3dnopad( PhiGrid, VGrid, h, nGridDims, m, d, np );
        printf("papa\n" );

        break;

      }


    // ~~~~~~~~~~ grid2scat


      switch (d) {

      case 1:
        g2s1d<<<32,256>>>(PhiScat,PhiGrid,y,nGridDim+2,n,d,m);

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

  }
