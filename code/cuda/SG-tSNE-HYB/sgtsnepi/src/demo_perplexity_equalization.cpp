/*!
  \file   demo_perplexity_equalization.cpp
  \brief  Conventional t-SNE usage.

  \author Dimitris Floros
  \date   2019-06-24
*/

#include <cmath>
#include <flann/flann.h>
#include <iostream>
#include <string>
#include <unistd.h>
#include <fstream>
using namespace std;
#include "sgtsne.hpp"
#include <sys/time.h>
//! Compute the approximate all-kNN graph of the input data points
/*!
  Compute the k-nearest neighbor dataset points of every point in
  the datsaet set using approximate k-nearest neighbor search (FLANN).

*/
void allKNNsearch(
    int *IDX,        //!< [k-by-N] array with the neighbor IDs
    double *DIST,    //!< [k-by-N] array with the neighbor distances
    double *dataset, //!< [L-by-N] array with coordinates of data points
    int N,           //!< [scalar] Number of data points N
    int dims,        //!< [scalar] Number of dimensions L
    int kappa) {     //!< [scalar] Number of neighbors k

  struct FLANNParameters p;

  p = DEFAULT_FLANN_PARAMETERS;
  p.algorithm = FLANN_INDEX_KDTREE;
  p.trees = 8;
  // p.log_level = FLANN_LOG_INFO;
    p.checks = 300;
  p.target_precision = 0.9;
//  p.checks = 300;

  // -------- Run a kNN search
  flann_find_nearest_neighbors_double(dataset, N, dims, dataset, N, IDX, DIST,
                                      kappa, &p);
}

int main(int argc, char **argv) {
  // ~~~~~~~~~~ variable declarations
  int opt;
  double u = 30;
  int sim, points;
  sim = 0;
  points=0;
  tsneparams params;
  std::string filename = "test.mtx";
  coord *y;

  // ~~~~~~~~~~ parse inputs

  // ----- retrieve the (non-option) argument:
  if ((argc <= 1) || (argv[argc - 1] == NULL) || (argv[argc - 1][0] == '-')) {
    // there is NO input...
    std::cerr << "No filename provided!" << std::endl;
    return 1;
  } else {
    // there is an input...
    filename = argv[argc - 1];
  }

  // ----- retrieve optional arguments

  // Shut GetOpt error messages down (return '?'):
  opterr = 0;
  char of[25]="timeInfo.txt";
  while ((opt = getopt(argc, argv, "d:a:m:e:h:p:u:s:n:f:")) != -1) {
    switch (opt) {
    case 'd':
      params.d = atoi(optarg);
      break;
    case 'm':
      params.maxIter = atoi(optarg);
      break;
    case 'e':
      params.earlyIter = atoi(optarg);
      break;
    case 'p':
      params.np = atoi(optarg);
      break;
    case 'u':
      sscanf(optarg, "%lf", &u);
      break;
    case 'a':
      sscanf(optarg, "%lf", &params.alpha);
      break;
    case 'h':
      sscanf(optarg, "%lf", &params.h);
      break;
    case 's':
      sscanf(optarg, "%d", &sim);
      break;
    case 'f':
	sscanf(optarg,"%s",of);
	std::cout<<of<<"\n";
	break;
    case 'n':
      sscanf(optarg, "%d", &points);
	break;
    case '?': // unknown option...
      std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
      break;
    }
  }

  int nn = std::ceil(u * 3);

  params.lambda = 1;

  // ~~~~~~~~~~ setup number of workers

  if (getWorkers() != params.np && params.np > 0)
    setWorkers(params.np);

  params.np = getWorkers();

  // ~~~~~~~~~~ read input data points
  int n, d;
  n=points;
  double * X;
  if(sim==0){
   X = readXfromMTX( filename.c_str(), &n, &d );
  }else if(sim>0){
   X =loadSimulationData<double>(n,50,sim);
   d=50;
  }else if(sim<0){
	d=50;
  X=load10x<coord>(n,50);

}
  params.n = n;

  // ~~~~~~~~~~ run kNN search

  std::cout << "Running k-neareast neighbor search for " << nn
            << " neighbors..." << std::flush;

  double *D = (double *)malloc(params.n * (nn + 1) * sizeof(double));
  int *I = (int *)malloc(params.n * (nn + 1) * sizeof(int));
/*
   ifstream Din;
   Din.open("D1m.txt");
   ifstream Iin;
        Iin.open("I1m.txt");
  for(int i=0;i<params.n * (nn + 1);i++){
        Din>>D[i];
        Iin>>I[i];
        }
        Din.close();
        Iin.close();
*/

  allKNNsearch(I, D, X, n, d, nn + 1);

 std::ofstream Dout;
 Dout.open("D1m.txt");
 std::ofstream Iout;
 Iout.open("I1m.txt");
 for(int i=0;i<params.n*(nn+1);i++){
	Iout<<I[i]<<"\n";
	Dout<<D[i]<<"\n";
  }
 Dout.close();
 Iout.close();

  std::cout << "DONE" << std::endl;

  sparse_matrix P = perplexityEqualization(I, D, n, nn, u);
    for(int i=0; i<91;i++){
        std::cout<<P.row[i]<<" "<<P.col[i]<<" "<<P.val[i]<<"\n";

        }

  free(D);
  free(I);

  params.n = n;
  d=params.d;
  double* timeInfo=(double *)malloc(sizeof(double)*8003);
  for(int i=0;i<8*1000+3;i++)timeInfo[i]=0;
  sim=0;
  if(sim==0){

   std::ifstream yinf;
   double *yin=(double*)malloc(sizeof(coord)*params.d*params.n);
  
   yinf.open("yin2m.txt");

   for(int i=0;i<n;i++){
     for(int j=0;j<d;j++){
     yinf>>yin[i*d+j];
   }
   }
   yinf.close();
   // ~~~~~~~~~~ Run SG-t-SNE
   struct timeval t1, t2;
  gettimeofday(&t1, NULL);
   y = sgtsne( P, params,yin,&timeInfo );
    gettimeofday(&t2, NULL);
  double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
  elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  elapsedTime/=1000.0;
  std::cout<<"sgtsnepi time= "<<elapsedTime<<"\n";
timeInfo[8*1000+2]=elapsedTime;

 free(yin);
 }else{
   struct timeval t1, t2;
  gettimeofday(&t1, NULL);

   y = sgtsne( P, params,nullptr,&timeInfo);
      gettimeofday(&t2, NULL);

  double elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0;   // us to ms
  elapsedTime/=1000.0;
  std::cout<<"sgtsnepi time= "<<elapsedTime<<"\n";
timeInfo[8*1000+2]=elapsedTime;

 }

ExtractTimeInfo(timeInfo,n,d,"timeInfo.txt");
/*
 std::cout << "timeInfo: N AttrTime S2G G2G G2S Phi2F Prep  SyncTime Totaltime Pi GDS: \n" ;
 for(int i=0;i<10;i++){
   std::cout << timeInfo[i]<<" "  ;
 }
std::cout << "\n";
*//*
 ofstream fout;  // Create Object of Ofstream
 ifstream fin;
 fin.open(of);
 fout.open (of,ios::app); // Append mode
 if(fin.is_open())
   {fout<<P.n<<" ";
     for(int i=0;i<10;i++){
       fout << timeInfo[i] << " ";
     }
     fout<<"\n";}
   fin.close();
   fout.close(); // Closing the file
*/

   // ~~~~~~~~~~ export results
   extractEmbeddingText( y, params.n, params.d );

  free(y);
}
