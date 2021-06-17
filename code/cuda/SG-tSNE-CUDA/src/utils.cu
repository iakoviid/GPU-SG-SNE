#include "utils.cuh"
#include <random>
#include <string>
#define N_GRID_SIZE 137
template <class dataPoint>
void appendProgressGPU(dataPoint *yd,int n,int d,const char* name){
 dataPoint *y=(dataPoint*)malloc(n*d*sizeof(dataPoint));
 gpuErrchk(cudaMemcpy(y, yd, n * d * sizeof(dataPoint),cudaMemcpyDeviceToHost));
  std::ofstream fout;  // Create Object of Ofstream
  std::ifstream fin;
  fin.open(name);
  fout.open (name,std::ios::app); // Append mode
   if (fin.is_open())
    {
      for (int i = 0 ; i < n ; i++ ){
        for (int j = 0 ; j < d ; j++ ){
          fout << y[i + j*n] << " ";
        }
        fout << std::endl;
      }


    }
   fin.close();
   fout.close();
free(y);



}
template <class dataPoint>
void savePtxt(int* csrrow,int* csrcol,dataPoint* csrval,int n,const char* name){
  std::ofstream f (name);
  if (f.is_open())
  {
    f<<n<<" "<<n<<" "<<csrrow[n]<<"\n";
    for (int i = 0 ; i < n ; i++ ){
      for (int element = csrrow[i] ;element < csrrow[i+1]; element++ ){
        f <<i <<" "<< csrcol[element] << " "<< csrval[element];
        f << std::endl;
      }
    }

    f.close();
  }
}
template<class dataPoint>
dataPoint* load10x(int n,int d){
  // ~~~~~~~~~~ variable declarations
  dataPoint *X;
  // ~~~~~~~~~~ read matrix
  // open the file
  std::ifstream fin( "data.txt" );
  // read defining parameters
  // allocate space for COO format
  X = static_cast<dataPoint *>( malloc( n* d* sizeof(dataPoint)) );
  // read the COO data
  for (int l = 0; l < n; l++){
  for (int j = 0; j < d; j++)
{fin >> X[l*d + j];
}
}
//  std::cout<<"Test --------------------------\n";
//  std::cout<<"1X:" <<X[1]<<" "<<X[2]<<"\n";
//  std::cout<<"2X:" <<X[50]<<" "<<X[51]<<"\n";

  // close connection to file
  fin.close();
  // ~~~~~~~~~~ return value
  return X;
}
void ExtractTimeInfo(double* timeInfo,int n,int d,const char* fname){
  std::ofstream fout;  // Create Object of Ofstream
  std::ifstream fin;
 double sum[7]={0};
  fin.open(fname);
  fout.open (fname,std::ios::app); // Append mode
  if(fin.is_open()){
    fout<<"N d gds matrix total: \n";
    fout<<n<<" "<< d<<" "<<timeInfo[7*1000]<<" "<<timeInfo[7*1000+1]<<" "<<timeInfo[7*1000+2]<<  "\n";
    fout<<"Fattr S2G G2G G2S Phi2F Nuconv Pre";
    for(int i=0;i<1000;i++){
      for(int j=0;j<7;j++){
        fout<<timeInfo[i*7 +j]<<" ";
        sum[j]+=timeInfo[i*7 +j];
      }
      fout<<"\n";
    }

  }
  fin.close();
  fout.close(); // Closing the file
  std::cout << "TimeAnalysis:" << '\n';
  std::cout << "N d GDS Matrix Total" << '\n';
  std::cout<<n<<" "<< d<<" "<<timeInfo[7*1000]/1000<<" "<<timeInfo[7*1000+1]/1000<<" "<<timeInfo[7*1000+2]/1000<<  "\n";
  std::cout<<"FAttr S2G G2G G2S Phi2F Nuconv Pre \n";
  for(int i=0;i<7;i++){
    std::cout<<sum[i]/1000<<" ";
  }
  std::cout<<"\n";

}
template <class dataPoint>
void extractEmbeddingText( dataPoint *y, int n, int d,const char* name ){

  std::ofstream f (name);

  if (f.is_open())
    {
      for (int i = 0 ; i < n ; i++ ){
        for (int j = 0 ; j < d ; j++ ){
          f << y[i*d + j] << " ";
        }
        f << std::endl;
      }

      f.close();
    }

}
template <class dataPoint>
void extractEmbeddingTextT( dataPoint *y, int n, int d,const char* name ){

  std::ofstream f (name);

  if (f.is_open())
    {
      for (int i = 0 ; i < n ; i++ ){
        for (int j = 0 ; j < d ; j++ ){
          f << y[i + j*n] << " ";
        }
        f << std::endl;
      }

      f.close();
    }

}
template <class dataPoint>
void extractEmbedding( dataPoint *y, int n, int d ){

  FILE * pFile;
  pFile = fopen ("embedding.bin", "wb");

  fwrite(y , sizeof(dataPoint), n*d, pFile);
  fclose(pFile);
  return;
}
void printParams(tsneparams P){

  std::cout << "Number of vertices: " << P.n << std::endl
            << "Embedding dimensions: " << P.d << std::endl
            << "Rescaling parameter λ: " << P.lambda << std::endl
            << "Early exag. multiplier α: " << P.alpha << std::endl
            << "Maximum iterations: " << P.maxIter << std::endl
            << "Early exag. iterations: " << P.earlyIter << std::endl
            << "Box side length h: " << P.h << std::endl
            << "Drop edges originating from leaf nodes? " << P.dropLeaf << std::endl
            << "Number of processes: " << P.np << std::endl;

}
template <class dataPoint>
dataPoint randn() {
  dataPoint x, y, radius;
  do {
    x = 2 * (rand() / ((dataPoint) RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((dataPoint) RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  y *= radius;
  return x;
}

template <class dataPoint> dataPoint *generateRandomCoord(int n, int d) {

  dataPoint *y = (dataPoint *)malloc(n * d * sizeof(dataPoint));

  for (int i = 0; i < n * d; i++)
    y[i] = ((dataPoint)rand() / (RAND_MAX)) * .0001;

  return y;
}
template <class dataPoint> dataPoint *generateRandomGaussianCoord(int n, int d) {
  std::default_random_engine generator;
  std::normal_distribution<dataPoint> distribution(0.0, 1.0);
  dataPoint *y = (dataPoint *)malloc(n * d * sizeof(dataPoint));

  for (int i = 0; i < n*d ; i++)y[i]=0.0001 * distribution(generator);

  return y;
}
int getBestGridSize(int nGrid) {

  // list of FFT sizes that work "fast" with FFTW
  int listGridSize[N_GRID_SIZE] = {
      8,   9,   10,  11,  12,  13,  14,  15,  16,  20,  25,  26,  28,  32,
      33,  35,  36,  39,  40,  42,  44,  45,  48,  49,  50,  52,  54,  55,
      56,  60,  63,  64,  65,  66,  70,  72,  75,  77,  78,  80,  84,  88,
      90,  91,  96,  98,  99,  100, 104, 105, 108, 110, 112, 117, 120, 125,
      126, 130, 132, 135, 140, 144, 147, 150, 154, 156, 160, 165, 168, 175,
      176, 180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 216, 220, 224,
      225, 231, 234, 240, 245, 250, 252, 260, 264, 270, 273, 275, 280, 288,
      294, 297, 300, 308, 312, 315, 320, 325, 330, 336, 343, 350, 351, 352,
      360, 364, 375, 378, 385, 390, 392, 396, 400, 416, 420, 432, 440, 441,
      448, 450, 455, 462, 468, 480, 490, 495, 500, 504, 512};

  // select closest (larger) size for given grid size
  for (int i = 0; i < N_GRID_SIZE; i++)
    if ((nGrid + 2) <= listGridSize[i])
      return listGridSize[i] - 2;

  return listGridSize[N_GRID_SIZE - 1] - 2;
}
template <class dataPoint>
dataPoint * readXfromMTX( const char *filename, int *n, int *d ){

  // ~~~~~~~~~~ variable declarations
  dataPoint *X;
  // ~~~~~~~~~~ read matrix

  // open the file
  std::ifstream fin( filename );
  std::cout<<filename<<"\n";
  // ignore headers and comments
//  while (fin.peek() == '%') fin.ignore(2048, '\n');

  int Npts=n[0];
  // read defining parameters
  fin >> n[0] >> d[0];
   std::cout<<n[0]<<"\n";
//  std::cout<<"n "<<n[0]<<" d "<<d[0]<<"\n";
  if(Npts>0){n[0]=Npts;}
  // allocate space for COO format
  X = static_cast<dataPoint *>( malloc( n[0] * d[0] * sizeof(dataPoint)) );


  // read the COO data
  for (int l = 0; l < n[0]; l++){
  for (int j = 0; j < d[0]; j++)
{
   fin >> X[l*d[0] + j];

}

}

  // close connection to file
  fin.close();
  // fout.close();

  // ~~~~~~~~~~ return value
  return X;

}
template <class dataPoint>
dataPoint* loadSimulationData(int n,int dim){


  std::default_random_engine generator;
  std::normal_distribution<dataPoint> distribution1(-10.0, 1.0);
  std::normal_distribution<dataPoint> distribution2(10.0, 1.0);
  dataPoint* h_X=(dataPoint*)malloc(n*dim*sizeof(dataPoint));

  for (int i = 0; i < dim * n; i ++) {
      if (i < ((n / 2) * dim)) {
          h_X[i] = distribution1(generator);
      } else {
          h_X[i] = distribution2(generator);
      }
  }
  return h_X;
}
template <class dataPoint>
dataPoint* loadSimulationData4(int n,int dim){


  std::default_random_engine generator;
  int n_clusters=4;
  std::normal_distribution<dataPoint> distribution1(-20.0, 2.0);
  std::normal_distribution<dataPoint> distribution2(20.0, 2.0);
  std::normal_distribution<dataPoint> distribution3(10.0, 2.0);
  std::normal_distribution<dataPoint> distribution4(-10.0, 2.0);

  dataPoint* h_X=(dataPoint*)malloc(n*dim*sizeof(dataPoint));

  for (int i = 0; i < n; i ++) {
    for(int d=0;d<dim;d++){

      if (i < ((n / n_clusters) )) {
          h_X[i*dim+d] = distribution1(generator);
      } else if(i < ((2*n / n_clusters) )) {
          h_X[i*dim+d] = distribution2(generator);
      }
      else if(i < ((3*n / n_clusters) ))
      {
         h_X[i*dim+d] = distribution3(generator);
     }
     else{
       h_X[i*dim+d] = distribution4(generator);

     }
}

  }
  return h_X;
}

template <class dataPoint>
dataPoint* loadSimulationData(int n,int dim,int n_clusters){


  std::default_random_engine generator;

  std::normal_distribution<dataPoint> distribution1(-20.0, 2.0);
  std::normal_distribution<dataPoint> distribution2(20.0, 2.0);

  dataPoint* h_X=(dataPoint*)malloc(n*dim*sizeof(dataPoint));
  for(int j=0;j<n_clusters;j++){
    for(int i=j*n/n_clusters;i<(j+1)*n/n_clusters;i++){
      for(int d=0;d<dim;d++){
        if(d>=j){
          h_X[i*dim+d] = distribution2(generator);

        }else{
          h_X[i*dim+d] = distribution1(generator);
        }
      }
    }
  }

  return h_X;
}
template <class dataPoint>
sparse_matrix<dataPoint> buildPFromMTX( const char *filename ){

  // ~~~~~~~~~~ variable declarations
  sparse_matrix<dataPoint> P;
  dataPoint *val_coo;
  matidx *row_coo, *col_coo;

  // ~~~~~~~~~~ read matrix

  // open the file
  std::ifstream fin( filename );

  // ignore headers and comments
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  // read defining parameters
  fin >> P.m >> P.n >> P.nnz;

  // allocate space for COO format
  val_coo = static_cast<dataPoint *>( malloc(P.nnz * sizeof(dataPoint)) );
  row_coo = static_cast<matidx *>( malloc(P.nnz * sizeof(matidx)) );
  col_coo = static_cast<matidx *>( malloc(P.nnz * sizeof(matidx)) );

  // read the COO data
  for (int l = 0; l < P.nnz; l++)
    fin >> row_coo[l] >> col_coo[l] >> val_coo[l];

  // close connection to file
  fin.close();

  // ~~~~~~~~~~ transform COO to CSC
  P.val = static_cast<dataPoint *>( malloc( P.nnz   * sizeof(matval)) );
  P.row = static_cast<matidx *>( malloc( P.nnz   * sizeof(matidx)) );
  P.col = static_cast<matidx *>( calloc( (P.n+1),  sizeof(matidx)) );

  // ----- find the correct column sizes
  for (int l = 0; l < P.nnz; l++){
    P.col[ col_coo[l]-1 ]++;
  }

  for(int i = 0, cumsum = 0; i < P.n; i++){
    int temp = P.col[i];
    P.col[i] = cumsum;
    cumsum += temp;
  }
  P.col[P.n] = P.nnz;

  // ----- copy the row indices to the correct place
  for (int l = 0; l < P.nnz; l++){
    int col = col_coo[l]-1;
    int dst = P.col[col];
    P.row[dst] = row_coo[l]-1;
    P.val[dst] = val_coo[l];

    P.col[ col ]++;
  }

  // ----- revert the column pointers
  for(int i = 0, last = 0; i < P.n; i++) {
    int temp = P.col[i];
    P.col[i] = last;

    last = temp;
  }

  // ~~~~~~~~~~ deallocate memory
  free( val_coo );
  free( row_coo );
  free( col_coo );

  // ~~~~~~~~~~ return value
  return P;

}
template float* load10x(int n,int d);
template double* load10x(int n,int d);

template void appendProgressGPU(float *yd,int n,int d,const char* name);
template
void appendProgressGPU(double *yd,int n,int d,const char* name);
template
sparse_matrix<float> buildPFromMTX( const char *filename );

template
sparse_matrix<double> buildPFromMTX( const char *filename );
template void extractEmbeddingText( float *y, int n, int d,const char* name );

template void extractEmbeddingTextT( float *y, int n, int d,const char* name );

template void extractEmbedding( float *y, int n, int d );

template void savePtxt(int* csrrow,int* csrcol,float* csrval,int n,const char* name);

template float randn();

template void extractEmbeddingText( double *y, int n, int d,const char* name );

template void extractEmbeddingTextT( double *y, int n, int d,const char* name );

template void extractEmbedding( double *y, int n, int d );

template void savePtxt(int* csrrow,int* csrcol,double* csrval,int n,const char* name);

template double randn();
template float * readXfromMTX( const char *filename, int *n, int *d );
template double * readXfromMTX( const char *filename, int *n, int *d );
template float* loadSimulationData(int n,int dim);
template double* loadSimulationData(int n,int dim);
template double* loadSimulationData(int n,int dim,int n_clusters);
template float* loadSimulationData(int n,int dim,int n_clusters);
template double *generateRandomCoord(int n, int d);
template float *generateRandomCoord(int n, int d);
template double *generateRandomGaussianCoord(int n, int d);
template float *generateRandomGaussianCoord(int n, int d);
