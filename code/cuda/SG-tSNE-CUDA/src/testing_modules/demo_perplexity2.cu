#include "../types.hpp"
#include "../utils.cuh"
#include "perplexity_search0.cuh"
#include "sgtsne.cuh"
#include <fstream>
#include <iostream>
#include <unistd.h>
#include <allknn.cuh>
int OcMaxBlockSize=512;
int GridSize=64;

#include <hybrid.cuh>
using namespace std;

int main(int argc, char **argv) {
  int opt;
  coord u = 30;
  coord *y;
  tsneparams params;
  std::string filename = "test.txt";
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

  while ((opt = getopt(argc, argv, "d:a:m:e:h:p:u:f:g:s:n:t:")) != -1) {
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
      sscanf(optarg, "%f", &u);
      break;
    case 'a':
      sscanf(optarg, "%f", &params.alpha);
      break;
    case 'h':
      sscanf(optarg, "%f", &params.h);
      break;
    case 'f':
      sscanf(optarg, "%d", &params.ComputeError);
      break;
    case 'g':
      sscanf(optarg, "%d", &params.ng);
      break;
    case 's':
      sscanf(optarg, "%d", &params.sim);
      break;
    case 'n':
      sscanf(optarg, "%d", &params.n);
      break;
    case 't':
      sscanf(optarg, "%d", &params.format);
      break;
    case '?': // unknown option...
      std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
      break;
    }
  }
  int nn = std::ceil(u * 3);
  params.lambda = 1;
  coord *X;
  int n, d;
  if (params.sim == 0) {
    n=params.n;
    X = readXfromMTX<coord>(filename.c_str(), &n, &d);
  } else if(params.sim>0) {
    n = params.n;
    X =loadSimulationData<coord>(n,50,params.sim);
    d = 50;
  }else if(params.sim<0){
	d=50;
	n=params.n;
	X=load10x<coord>(n,50);
}
  params.n = n;
  // ~~~~~~~~~~ run kNN search
  std::cout<<"Number of points "<< n<<" "<<"dimensions "<<d<<"\n";
  std::cout << "Running k-neareast neighbor search for " << nn
            << " neighbors..." << std::flush;

  coord *D = (coord *)malloc(params.n * (nn + 1) * sizeof(coord));
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
allKNNsearchflann(I, D, X, n, d, nn + 1);
/*

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
*/

  int *Id;
  CUDA_CALL(cudaMallocManaged(&Id, params.n * (nn + 1) * sizeof(int)));
  CUDA_CALL(cudaMemcpy(Id, I, params.n * (nn + 1) * sizeof(int),
                       cudaMemcpyHostToDevice));

  free(I);
  /*Max normalize D*/
cudaDeviceSynchronize();

 coord max=0;
  for(int i=0;i<params.n * (nn + 1);i++)if(D[i]>max){max=D[i];}
 for(int i=0;i<params.n * (nn + 1);i++)D[i]/=max;

  std::cout << "DONE" << std::endl;
  sparse_matrix<coord> P = perplexityEqualization(Id, D, n, nn, u);
  free(D);
  CUDA_CALL(cudaFree(Id));

  params.n = n;
  d = params.d;
  // ~~~~~~~~~~ Run SG-t-SNE
  struct timeval t1, t2;

  double timeInfo[7*1000+3] = {0};
  double elapsedTime;

  int blockSize;   // The launch configurator returned block size 
  int minGridSize; // The minimum grid size needed to achieve the 

  cudaOccupancyMaxPotentialBlockSize( &minGridSize, &blockSize,ell_spmv_kernel<float,2>, 0, 0);
   std::cout<<"gridSize: "<<minGridSize<<" and blockSize for maximal Occupancy: "<<blockSize<<"\n";
   
   int deviceCount = 0;
   cudaGetDeviceCount(&deviceCount);
   int numSMs;
   cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, 0);
   std::cout<<"Number of SMs: "<<numSMs<<" BlockSize for maximal Occupancy: "<<blockSize<<"\n";
   OcMaxBlockSize=blockSize;
   GridSize=16*numSMs;
   GridSize=minGridSize;  
// cudaOccupancyMaxPotentialBlockSizeVariableSMemWithFlags ( &minGridSize, &blockSize, ell_spmv_kernel<float,2> );
  // std::cout<<"gridSize: "<<minGridSize<<" and blockSize for maximal Occupancy: "<<blockSize<<"\n";

if (params.sim == 0) {
    coord *yin = (coord *)malloc(sizeof(coord) * d * n);
    ifstream yinf;

    
      yinf.open("yin2m.txt");
    
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < d; j++) {
        yinf >> yin[i + j * n];
      }
    }
    yinf.close();
    gettimeofday(&t1, NULL);

    y = sgtsneCUDA<coord>(&P, params,yin, timeInfo);
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    printf("Total time=%lf\n",elapsedTime );
    free(yin);
  } else {
    gettimeofday(&t1, NULL);
    y = sgtsneCUDA<coord>(&P, params, NULL, timeInfo);
    gettimeofday(&t2, NULL);
    elapsedTime = (t2.tv_sec - t1.tv_sec) * 1000.0;    // sec to ms
    elapsedTime += (t2.tv_usec - t1.tv_usec) / 1000.0; // us to ms
    printf("Total time=%lf\n",elapsedTime );

  }
  timeInfo[7*1000+2]=elapsedTime;
  ExtractTimeInfo(timeInfo,n,d,"timeInfo.txt");
/*
  ofstream fout;  // Create Object of Ofstream
  ifstream fin;

  fin.open("timeGPU.txt");
  fout.open ("timeGPU.txt",ios::app); // Append mode
  if(fin.is_open())
    { fout<<n<<" ";
      for(int i=0;i<10;i++){fout<<timeInfo[i]/1000<<" ";}
      fout<<"\n";
    }
  fin.close();
  fout.close(); // Closing the file

  std::cout << "TimeAnalysis:" << '\n';
  std::cout << "N FAttr S2G G2G G2S Phi2F Nuconv Pre GDS Matrix Total" << '\n';
  cout<<n<<" ";
  for(int i=0;i<10;i++){cout<<timeInfo[i]/1000<<" ";}
  cout<<"\n";
*/
  // ~~~~~~~~~~ export results
  extractEmbeddingTextT(y, params.n, params.d, "sgtsneEmbedding.txt");
  free(y);

  return 0;
}
