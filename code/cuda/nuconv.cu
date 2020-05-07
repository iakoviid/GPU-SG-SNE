
void nuconv( double *PhiScat, double *y, double *VScat,
             uint32_t *ib, uint32_t *cb,
             int n, int d, int m, int np, int nGridDim,
             double* y_d,uint32_t* id_d, uint32_t* cb_d, double* VScat_d, double* Phi_d){


  // ~~~~~~~~~~ normalize coordinates (inside bins)
  double maxy = 0;
  int Blocks=64;
  int threads=1024;
  for (int i = 0; i < n*d; i++)
    maxy = maxy < y[i] ? y[i] : maxy;
  for(int i=0 ;i<n*d;i++){y[i] /= maxy;}


  thrust::device_ptr<double> yVec_ptr(y_d);
  thrust::device_vector<double> yVec_d(yVec_ptr,yVec_ptr+n*d);
  thrust::device_vector<double>::iterator iter=thrust::max_element(yVec_d.begin(),yVec_d.end());
  unsigned int position= iter-yVec_d.begin();
   maxy=yVec_d[position];
  // ~~~~~~~~~~ scale them from 0 to ng-1

  for(int i=0;i<n*d;i++){
    if(y[i]==1){
      y[i] = y[i] - std::numeric_limits<double>::epsilon();

    }
    y[i] *= (nGridDim-1);

  }

  for (int i = 0; i< n*d; i++)
    if ( (y[i] >= nGridDim-1) || (y[i] < 0) ) exit(1);

  // ~~~~~~~~~~ find exact h

  double h = maxy / (nGridDim - 1 - std::numeric_limits<double>::epsilon() );


  // ~~~~~~~~~~ scat2grid
  int szV = pow( nGridDim+2, d ) * m;
  printf("m=%d d=%d\n",m,d );
  double *VGrid = static_cast<double *> ( calloc( szV * np, sizeof(double) ) );
  double* VGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, pow( nGridDim+2, d )*(d+1)*sizeof(double)));
  double* V=(double *)( calloc( szV * np, sizeof(double) ) );
  double *VScat2    = (double*) malloc( n*(d+1) * sizeof( double ) );

  CUDA_CALL(cudaMemcpy(VScat2,VScat_d,n*(d+1) * sizeof( double ), cudaMemcpyDeviceToHost));
  printf("=====================================================================================\n" );
  for(int i=0;i<n;i++){
    for(int j=0;j<d+1;j++){
      if(abs(VScat[i*(d+1)+j] -VScat2[i+j*n])<0.01){
        //printf(" Succes VScat= %lf VScat_d=%lf\n",VScat[i*(d+1)+j],VScat2[i+j*n] );
      }else{
        printf(" Error VScat= %lf VScat_d=%lf\n",VScat[i*(d+1)+j],VScat2[i+j*n] );}
  }}
  printf("===================================================================================\n" );
  int tpoints=pow( nGridDim+2, d );

  if(d==1){
    s2g1d( VGrid, y, VScat, nGridDim+2, np, n, d, m );
    s2g1dCuda<<<1,1>>>(VGrid_d,y_d,VScat_d,nGridDim+2,n,d,m,maxy);
    //Differnt strategy every point must take 1 f1

    CUDA_CALL(cudaMemcpy(V,VGrid_d,szV*sizeof(double), cudaMemcpyDeviceToHost));

    for(int i=0;i<pow( nGridDim+2, d );i++){
      for(int j=0;j<m;j++){
        if(abs(VGrid[i+j*tpoints]- V[i+j*tpoints])<0.00001){
        //printf("Succes V1=%lf vs V=%lf \n",VGrid[i+j*tpoints] , V[i+j*tpoints]);
      }else{
      printf("Error V1=%lf vs V=%lf \n",VGrid[i+j*tpoints] , V[i+j*tpoints]);

    }
  }

  }

  printf("===================================================================================\n" );




  }

  double *PhiGrid = static_cast<coord *> ( calloc( szV, sizeof(double) ) );
  double *PhiGrid_d;
  CUDA_CALL(cudaMallocManaged(&VGrid_d, szV*sizeof(double)));

  uint32_t * const nGridDims = new uint32_t [d]();
  for(int i=0;i<d;i++){
    nGridDims[i] = nGridDim + 2;}
  if(d==1){
    conv1dnopad( PhiGrid, VGrid, h, nGridDims, m, d, 1 );
    
  }



}
