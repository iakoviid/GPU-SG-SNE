/*!
  \file   utils.cpp
  \brief  Auxilliary utilities.

  \author Dimitris Floros
  \date   2019-06-20
*/

#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <random>
#include <cilk/cilk_api.h>

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

void ExtractTimeInfo(double* timeInfo,int n,int d,const char* fname){
  std::ofstream fout;  // Create Object of Ofstream
  
  int length=8;
  int maxIter=1000;
  fout.open (fname); // Append mode
  double sum[8]={0};
  
    fout<<"N d gds matrix total";
    fout<<n<<" "<< d<<" "<<timeInfo[length*maxIter]<<" "<<timeInfo[length*maxIter+1]<<" "<<timeInfo[length*maxIter+2]<<  "\n";
    fout<<"Fattr S2G G2G G2S Phi2F Nuconv Pre Synch";
    for(int i=0;i<maxIter;i++){
      for(int j=0;j<length;j++){
        fout<<timeInfo[i*length +j]<<" ";
        sum[j]+=timeInfo[i*length +j];
      }
      fout<<"\n";
    }

  
 
  fout.close(); // Closing the file
  std::cout << "TimeAnalysis:" << '\n';
  std::cout << "N d GDS Matrix Total" << '\n';
  std::cout<<n<<" "<< d<<" "<<timeInfo[length*maxIter]<<" "<<timeInfo[length*maxIter+1]<<" "<<timeInfo[length*maxIter+2]<<  "\n";
  std::cout<<"FAttr S2G G2G G2S Phi2F Nuconv Pre Synch \n";
  for(int i=0;i<length;i++){
    std::cout<<sum[i]<<" ";
  }
  std::cout<<"\n";

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
template float* load10x(int n,int d);
template double* load10x(int n,int d);

double randn() {
  double x, y, radius;
  do {
    x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  y *= radius;
  return x;
}

sparse_matrix buildPFromMTX( const char *filename ){

  // ~~~~~~~~~~ variable declarations
  sparse_matrix P;
  matval *val_coo;
  matidx *row_coo, *col_coo;

  // ~~~~~~~~~~ read matrix

  // open the file
  std::ifstream fin( filename );

  // ignore headers and comments
  while (fin.peek() == '%') fin.ignore(2048, '\n');

  // read defining parameters
  fin >> P.m >> P.n >> P.nnz;

  // allocate space for COO format
  val_coo = static_cast<matval *>( malloc(P.nnz * sizeof(matval)) );
  row_coo = static_cast<matidx *>( malloc(P.nnz * sizeof(matidx)) );
  col_coo = static_cast<matidx *>( malloc(P.nnz * sizeof(matidx)) );

  // read the COO data
  for (int l = 0; l < P.nnz; l++)
    fin >> row_coo[l] >> col_coo[l] >> val_coo[l];

  // close connection to file
  fin.close();

  // ~~~~~~~~~~ transform COO to CSC
  P.val = static_cast<matval *>( malloc( P.nnz   * sizeof(matval)) );
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


void extractEmbeddingText( double *y, int n, int d ){

  std::ofstream f ("embedding.txt");

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


void extractEmbedding( double *y, int n, int d ){

  FILE * pFile;
  pFile = fopen ("embedding.bin", "wb");

  fwrite(y , sizeof(double), n*d, pFile);
  fclose(pFile);
  return;
}

void setWorkers(int n){
  std::string str = std::to_string(n);

  __cilkrts_end_cilk();
  if ( 0!=__cilkrts_set_param("nworkers", str.c_str() ) )
    std::cerr << "Error setting workers" << std::endl;
}

int getWorkers(){
  return __cilkrts_get_nworkers();
}
#include <string>
double * readXfromMTX( const char *filename, int *n, int *d ){

  // ~~~~~~~~~~ variable declarations
  double *X;
  // ~~~~~~~~~~ read matrix

  // open the file
  std::ifstream fin( filename );

  // ignore headers and comments
  while (fin.peek() == '%') fin.ignore(2048, '\n');
  int Npts=n[0];
  // read defining parameters
  fin >> n[0] >> d[0];
  if(Npts>0){n[0]=Npts;}
  // allocate space for COO format
  X = static_cast<double *>( malloc( n[0] * d[0] * sizeof(matval)) );

  // read the COO data
  for (int l = 0; l < n[0]; l++){
//  std::string word;
 // fin >> word;
for (int j = 0; j < d[0]; j++)
    {  fin >> X[l*d[0] + j];}
}

  // close connection to file
  fin.close();
 /*
 std::ofstream fout;
  fout.open("glove2M.txt");
  fout << n[0]<<" "<< d[0]<<"\n";
   for (int l = 0; l < n[0]; l++){
    for (int j = 0; j < d[0]; j++)
    {  fout << X[l*d[0] + j]<<" ";}
	fout<< "\n";
}
fout.close();
*/
 // ~~~~~~~~~~ return value
  return X;

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
// ##################################################
// FUNCTIONS IMPLEMENTED BY VAN DER MAATEN TO INPUT/OUTPUT DATA
// IN THE SAME FORMAT

// Function that loads data from a t-SNE file
// Note: this function does a malloc that should be freed elsewhere
bool vdm_load_data(double** data, int* n, int* d, int* no_dims, double* theta, double* perplexity, int* rand_seed, int* max_iter) {

  // Open file, read first 2 integers, allocate memory, and read the data
  FILE *h;
  if((h = fopen("data.dat", "r+b")) == NULL) {
    printf("Error: could not open data file.\n");
    return false;
  }
  fread(n, sizeof(int), 1, h);             // number of datapoints
  fread(d, sizeof(int), 1, h);             // original dimensionality
  fread(theta, sizeof(double), 1, h);      // gradient accuracy
  fread(perplexity, sizeof(double), 1, h); // perplexity
  fread(no_dims, sizeof(int), 1, h);       // output dimensionality
  fread(max_iter, sizeof(int),1,h);        // maximum number of iterations
  *data = (double*) malloc(*d * *n * sizeof(double));
  if(*data == NULL) { printf("Memory allocation failed!\n"); exit(1); }
  fread(*data, sizeof(double), *n * *d, h);         // the data
  if(!feof(h)) fread(rand_seed, sizeof(int), 1, h); // random seed
  fclose(h);
  printf("Read the %i x %i data matrix successfully!\n", *n, *d);
  return true;
}

// Function that saves map to a t-SNE file
void vdm_save_data(double* data, int* landmarks, double* costs, int n, int d) {

  // Open file, write first 2 integers and then the data
  FILE *h;
  if((h = fopen("result.dat", "w+b")) == NULL) {
    printf("Error: could not open data file.\n");
    return;
  }
  fwrite(&n, sizeof(int), 1, h);
  fwrite(&d, sizeof(int), 1, h);
  fwrite(data, sizeof(double), n * d, h);
  fwrite(landmarks, sizeof(int), n, h);
  fwrite(costs, sizeof(double), n, h);
  fclose(h);
  printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}
template double* loadSimulationData(int n,int dim,int n_clusters);
template float* loadSimulationData(int n,int dim,int n_clusters);
