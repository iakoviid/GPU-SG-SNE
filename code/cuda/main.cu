/*Main Function input data*/

#include <stdio.h>
#include <stdlib.h>
#include "gradient_descend.cpp"



double *generateRandomCoord(int n, int d)
 {

  double *y = (double *)malloc(n * d * sizeof(double));
  srand(time(0));

  for (int i = 0; i < n * d; i++)
    y[i] = ((double) rand() / (RAND_MAX))*100;

  return y;
}

void kl_minimization(coord* y,
                     tsneparams params,
                     BiCsb<matval, matidx> *csb,
                     double **timeInfo = nullptr)
										 {

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int    stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta    = 200.0;
  int    iterPrint = 50;

  double timeFattr = 0.0;
  double timeFrep  = 0.0;


  int    n = params.n;
  int    d = params.d;
  int    max_iter = params.maxIter;

  coord zeta = 0;

  // ----- Allocate memory
  coord* dy    = (coord*) malloc( n * d * sizeof(coord));
  coord* uy    = (coord*) malloc( n * d * sizeof(coord));
  coord* gains = (coord*) malloc( n * d * sizeof(coord));

  // ------ Initialize
  for(int i = 0; i < n*d; i++){
    uy[i] =  .0;
    gains[i] = 1.0;
  }

  // ----- Print precision
  if (sizeof(y[0]) == 4)
    std::cout << "Working with single precision" << std::endl;
  else if (sizeof(y[0]) == 8)
    std::cout << "Working with double precision" << std::endl;

  // ----- Start t-SNE iterations
  for(int iter = 0; iter < max_iter; iter++) {

    // ----- Gradient calculation
    zeta = compute_gradient(dy, &timeFrep, &timeFattr, params, y, csb);

    // ----- Position update
    update_positions(dy, uy, n, d, y, gains, momentum, eta);

    // Stop lying about the P-values after a while, and switch momentum
    if(iter == stop_lying_iter) {
      params.alpha = 1;
    }

    // Change momentum after a while
    if(iter == mom_switch_iter){
      momentum = final_momentum;
    }

    // Print out progress
    if( iter % iterPrint == 0 || iter == max_iter - 1 ) {
      matval C = tsne_cost( csb, y, n, d, params.alpha, zeta );
      if(iter == 0){
        std::cout << "Iteration " << iter+1
                  << ": error is " << C
                  << std::endl;

      } else {
        double iterTime = tsne_stop_timer("QQ", start);
        std::cout << "Iteration " << iter
                  << ": error is " << C
                  << " (50 iterations in " << iterTime
                  << " seconds)"
                  << std::endl;

        start = tsne_start_timer();
      }
    }

  }

  // ----- Print statistics (time spent at PQ and QQ)
  std::cout << " --- Time spent in each module --- \n" << std::endl;
  std::cout << " Attractive forces: " << timeFattr
            << " sec [" << timeFattr / (timeFattr + timeFrep) * 100
            << "%] |  Repulsive forces: " << timeFrep
            << " sec [" << timeFrep / (timeFattr + timeFrep) * 100
            << "%]" << std::endl;


  free(dy);
  free(uy);
  free(gains);
}


int main(int argc,char ** argv){
  if(argc<2){
    printf("Put arguments [mode] [input_file]\n", );
    exit(1);
  }
  int mode=atoi(argv[1]);
  if(mode==0){
    n=atoi(argv[2]);
    d=atoi(argv[3]);
    s=atoi(argv[4]);
    printf("%d %d dimensional Random points in 0-100\n",n,d );
    printf("Starting Embedding in %d dimension \n",s );
    kl_minimization(y, params,BiCsb<matval, matidx> *csb,timeInfo = nullptr)


  }

    return 0;
}
