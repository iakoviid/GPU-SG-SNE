#include "tsne.cuh"
void tsneRunCpu(coord* y,coord* x,uint N,uint d,uint32_t Dim,uint32_t perplexity){
  tsneparams params;
  params.d         = d;         //!< Number of embedding dimensions
  params.lambda    = 1;         //!< λ rescaling parameter
  params.alpha     = 12;        //!< Early exaggeration multiplier
  params.maxIter   = 1000;      //!< Maximum number of iterations
  params.earlyIter = 250;       //!< Number of early exaggeration iterations
  params.n         = N;         //!< Number of vertices
  params.h         = -1;        //!< Grid side length (accuracy control)
  params.dropLeaf  = false;     //!< Drop edges originating from leaf nodes?
  params.np        = 0;         //!< Number of CILK workers (processes)
  double* timeInfo;



  kl_minimizationCPU(y,params,&timeInfo);

}
