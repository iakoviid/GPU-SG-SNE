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
                           tsneparams params, dataPoint *y, sparse_matrix P) {
  double timeInfo[7];
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
  pq(Fattr, y, P.val, P.row, P.col, n, d);
  *timeFattr = tsne_stop_timer("PQ", start);
  double sum = 0;
  for (int i = 0; i < n * d; i++) {
    sum += Fattr[i];
  }
  start = tsne_start_timer();
  double zeta =
      computeFrepulsive_interpCPU(Frep, y, n, d, params.h, 1, timeInfo);

  *timeFrep = tsne_stop_timer("QQ", start);
  printf("zeta=%lf sum=%lf\n", zeta, sum);

  // ----- Compute gradient (dY)
  compute_dyCPU(dy, Fattr, Frep, n, d, params.alpha);

  // ----- Free-up memory
  free(Fattr);
  free(Frep);
  return zeta;
}
void kl_minimizationCPU(coord *y, tsneparams params, sparse_matrix P) {

  // ----- t-SNE hard coded parameters - Same as in vdM's code
  int stop_lying_iter = params.earlyIter, mom_switch_iter = 250;
  double momentum = .5, final_momentum = .8;
  double eta = 200.0;
  int iterPrint = 50;

  double timeFattr = 0.0;
  double timeFrep = 0.0;

  // struct timeval start;

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

  for (int iter = 0; iter < max_iter; iter++) {
    printf("%d--------------------------------------\n", iter);

    zeta = compute_gradientCPU(dy, &timeFrep, &timeFattr, params, y, P);

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
  }

  free(dy);
  free(uy);
  free(gains);
}
