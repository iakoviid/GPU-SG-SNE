#include "../types.hpp"
#include "../utils.cuh"
#include "perplexity_search0.cuh"
#include "sgtsne.cuh"
#include <flann/flann.h>
#include <fstream>
#include <iostream>
#include <unistd.h>
using namespace std;

//! Compute the approximate all-kNN graph of the input data points
/*!
  Compute the k-nearest neighbor dataset points of every point in
  the datsaet set using approximate k-nearest neighbor search (FLANN).

*/
void allKNNsearch(int * IDX,        //!< [k-by-N] array with the neighbor IDs
                  float * DIST,    //!< [k-by-N] array with the neighbor distances
                  float * dataset, //!< [L-by-N] array with coordinates of data points
                  int N,            //!< [scalar] Number of data points N
                  int dims,         //!< [scalar] Number of dimensions L
                  int kappa) {      //!< [scalar] Number of neighbors k


  struct FLANNParameters p;

  p = DEFAULT_FLANN_PARAMETERS;
  p.algorithm = FLANN_INDEX_KDTREE;
  p.trees = 8;
  // p.log_level = FLANN_LOG_INFO;
  p.target_precision=0.9999999;
  p.checks = 100;

  // -------- Run a kNN search
  flann_find_nearest_neighbors_float(dataset, N, dims, dataset, N, IDX, DIST, kappa, &p);

}
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
    X = readXfromMTX<coord>(filename.c_str(), &n, &d);
  } else {
    n = params.n;
    X =loadSimulationData<coord>(n,50,params.sim);
    d = 50;
  }
  params.n = n;
  // ~~~~~~~~~~ run kNN search

  std::cout << "Running k-neareast neighbor search for " << nn
            << " neighbors..." << std::flush;

  coord *D = (coord *)malloc(params.n * (nn + 1) * sizeof(coord));
  int *I = (int *)malloc(params.n * (nn + 1) * sizeof(int));

  allKNNsearch(I, D, X, n, d, nn + 1);

  int *Id;
  CUDA_CALL(cudaMallocManaged(&Id, params.n * (nn + 1) * sizeof(int)));
  CUDA_CALL(cudaMemcpy(Id, I, params.n * (nn + 1) * sizeof(int),
                       cudaMemcpyHostToDevice));

  free(I);
  /*Max normalize D*/
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
  if (params.sim == 0) {
    coord *yin = (coord *)malloc(sizeof(coord) * d * n);
    ifstream yinf;
    if (d == 2) {
      yinf.open("yin2d.txt");
    } else if (d == 3) {
      yinf.open("yin3d.txt");
    }
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < d; j++) {
        yinf >> yin[i + j * n];
      }
    }
    yinf.close();

    y = sgtsneCUDA<coord>(&P, params, yin, NULL);
    free(yin);
  } else {
    y = sgtsneCUDA<coord>(&P, params, NULL, NULL);
  }
  // ~~~~~~~~~~ export results
  extractEmbeddingTextT(y, params.n, params.d, "sgtsneEmbedding.txt");
  free(y);

  return 0;
}
