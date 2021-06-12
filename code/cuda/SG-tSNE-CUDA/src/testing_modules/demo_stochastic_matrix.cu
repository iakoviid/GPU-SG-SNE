#include "common.cuh"
#include <iostream>
#include <stdio.h>
#include <sys/time.h>
using namespace std;
#include "../types.hpp"
#include "../sgtsne.cuh"
#include "../utils.cuh"
#include <fstream>
#include <unistd.h>

int main(int argc, char **argv) {
  //srand(time(NULL));
  coord u = 30;

  int opt;
  tsneparams params;
  std::string filename = "test.mtx";

  // ----- retrieve the (non-option) argument:
  if ( (argc <= 1) || (argv[argc-1] == NULL) || (argv[argc-1][0] == '-') ) {
    // there is NO input...
    std::cerr << "No filename provided!" << std::endl;
    return 1;
  }
  else {
    // there is an input...
    filename = argv[argc-1];
  }

  // ----- retrieve optional arguments

  // Shut GetOpt error messages down (return '?'):
  opterr = 0;

  while ( (opt = getopt(argc, argv, "d:a:m:e:h:p:u:f:g:")) != -1 ) {
    switch ( opt ) {
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
    case '?':  // unknown option...
      std::cerr << "Unknown option: '" << char(optopt) << "'!" << std::endl;
      break;
    }
  }
  sparse_matrix<coord> P = buildPFromMTX<coord>( filename.c_str() );
  params.n = P.m;
  int n=params.n;
  int d=params.d;
  coord *yin=(coord*)malloc(sizeof(coord)*d*n);
  ifstream yinf;

  if(d==2){
  yinf.open("yin2d.txt");
}else if(d==3){
   yinf.open("yin3d.txt");
}
  for(int i=0;i<n;i++){
    for(int j=0;j<d;j++){
    yinf>>yin[i+j*n];
  }
  }
  yinf.close();

  coord* y =sgtsneCUDA<coord>( &P,  params,yin,NULL);

  extractEmbeddingTextT(y, params.n, params.d, "sgtsneEmbedding.txt");
  free(yin);
  free( y );


return 0;
}
