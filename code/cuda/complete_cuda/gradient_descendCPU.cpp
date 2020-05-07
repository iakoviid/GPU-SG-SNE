#include "gradient_descend.hpp"



template <class dataPoint>
void compute_dyCPU(dataPoint *const dy, dataPoint const *const Fattr,
                   dataPoint const *const Frep, int const N, int const dim,
                   dataPoint const alpha) {

  for (int i = 0; i < N; i++) {
    for (int d = 0; d < dim; d++) {
      dy[i * dim + d] = (alpha * Fattr[i * dim + d] - Frep[i * dim + d]);
    }
  }
}

template <class dataPoint>
void update_positionsCPU(dataPoint *const dY, dataPoint *const uY, int const N,
                         int const no_dims, dataPoint *const Y,
                         dataPoint *const gains, double const momentum,
                         double const eta) {

  // Update gains
  for (int i = 0; i < N * no_dims; i++) {
    gains[i] = (sign(dY[i]) != sign(uY[i])) ? (gains[i] + .2) : (gains[i] * .8);
    if (gains[i] < .01)
      gains[i] = .01;
    uY[i] = momentum * uY[i] - eta * gains[i] * dY[i];
    Y[i] = Y[i] + uY[i];
  }
  // find mean
  dataPoint meany[no_dims];
  for (int i = 0; i < no_dims; i++) {
    meany[i] = 0;
  }
  for (int i = 0; i < no_dims; i++) {
    for (int j = 0; j < N; j++) {
      meany[i] += Y[j * no_dims + i];
    }


    meany[i] /= N;
  }

  // zero-mean
  for (int n = 0; n < N; n++) {
    for (int d = 0; d < no_dims; d++) {
      Y[n * no_dims + d] -= meany[d];
    }
  }
}
template <class dataPoint>
double compute_gradientCPU(dataPoint *dy, double *timeFrep, double *timeFattr,
                           tsneparams params, dataPoint *y) {

  // ----- parse input parameters
  int d = params.d;
  int n = params.n;

  // ----- timing
  struct timeval start;

  // ----- Allocate memory
  dataPoint *Fattr = (dataPoint *)calloc(n * d, sizeof(dataPoint));
  dataPoint *Frep = (dataPoint *)calloc(n * d, sizeof(dataPoint));

  // ------ Compute PQ (fattr)
  start = tsne_start_timer();
  // csb_pq( NULL, NULL, csb, y, Fattr, n, d, 0, 0, 0 );
  double zeta;
  /*
if (timeInfo != nullptr)
zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np,
                                       &timeInfo[1]);
else
zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np);
  */
  printf("HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH=%lf\n",params.h );
  zeta = computeFrepulsive_interp(Frep, y, n, d, params.h, params.np);

  *timeFrep += tsne_stop_timer("QQ", start);
  // double zeta = computeFrepulsive_exact(Frep, y, n, d);

  // ----- Compute gradient (dY)
  compute_dyCPU(dy, Fattr, Frep, n, d, params.alpha);

  // ----- Free-up memory
  free(Fattr);
  free(Frep);
  return zeta;
}
void kl_minimizationCPU(coord *y, tsneparams params, double **timeInfo = NULL) {

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;
  int iterPrint = 50;

  double timeFattr = 0.0;
  double timeFrep = 0.0;

  struct timeval start;

  int n = params.n;
  int d = params.d;
  int max_iter = params.maxIter;

  coord zeta = 0;

  // ----- Allocate memory
  coord *dy = (coord *)malloc(n * d * sizeof(coord));
  coord *uy = (coord *)malloc(n * d * sizeof(coord));
  coord *gains = (coord *)malloc(n * d * sizeof(coord));

  // ------ Initialize
  for (int i = 0; i < n * d; i++) {
    uy[i] = .0;
    gains[i] = 1.0;
  }

  // ----- Start t-SNE iterations
  start = tsne_start_timer();
  max_iter = 1;
  for (int iter = 0; iter < max_iter; iter++) {

    // ----- Gradient calculation
    if (timeInfo == NULL)
      zeta = compute_gradientCPU(dy, &timeFrep, &timeFattr, params, y);

    // ----- Position update
    update_positionsCPU(dy, uy, n, d, y, gains, momentum, eta);

    // Stop lying about the P-values after a while, and switch momentum
    if (iter == stop_lying_iter) {
      params.alpha = 1;
    }

    // Change momentum after a while
    if (iter == mom_switch_iter) {
      momentum = final_momentum;
    }

    // Print out progress
    if (iter % iterPrint == 0 || iter == max_iter - 1) {
      // matval C = tsne_cost( csb, y, n, d, params.alpha, zeta );
      if (iter == 0) {
        std::cout << "Iteration "
                  << iter + 1
                  //<< ": error is " << C
                  << std::endl;

      } else {
        double iterTime = tsne_stop_timer("QQ", start);
        std::cout << "Iteration "
                  << iter
                  //<< ": error is " << C
                  << " (50 iterations in " << iterTime << " seconds)"
                  << std::endl;

        start = tsne_start_timer();
      }
    }
  }

  // ----- Print statistics (time spent at PQ and QQ)
  std::cout << " --- Time spent in each module --- \n" << std::endl;
  std::cout << " Attractive forces: " << timeFattr << " sec ["
            << timeFattr / (timeFattr + timeFrep) * 100
            << "%] |  Repulsive forces: " << timeFrep << " sec ["
            << timeFrep / (timeFattr + timeFrep) * 100 << "%]" << std::endl;

  free(dy);
  free(uy);
  free(gains);
}
