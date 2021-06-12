/*!
  \file   graph_rescaling.cpp
  \brief  Routines regarding lambda-based graph rescaling.

  \author Dimitris Floros
  \date   2019-06-21
*/



#include "graph_rescaling.hpp"

#include <limits>
#include <cmath>

#include <iostream>
template<typename dataPoint>
void lambdaRescaling( sparse_matrix<dataPoint> P, dataPoint lambda, bool dist, bool dropLeafEdge ){

  dataPoint tolBinary   = 1e-5;
  int    maxIter     = 100;
  dataPoint *sig2       = static_cast<dataPoint *>( malloc(P.n*sizeof(dataPoint)) );

  if (dist)  std::cout << "Input considered as distances" << std::endl;

  for (int i=0; i<P.n; i++){

    dataPoint fval = 1 - lambda;
    sig2[i] = 1;
    dataPoint a = -1e3;
    dataPoint c = 1e7;

    int iter = 0;
    dataPoint sum_i = 0;

    if (!dist)  // transform values to distances

      for (uint32_t j=P.col[i]; j < P.col[i+1]; j++)
        P.val[j] = -log( P.val[j] );

    else{  // distances are given, find starting summation

      for (uint32_t j=P.col[i]; j < P.col[i+1]; j++) {
        sum_i += exp( -P.val[j] );
      }

      fval = sum_i - lambda;

    }


    // bisection search
    while (std::abs( fval ) > tolBinary && iter < maxIter) {

      //printf("sigma[%d]=%lf sum_i=%lf\n",i,sig2[i],sum_i );
      iter++; sum_i = 0;

      if (fval > 0) {
        a = sig2[i];

        //if (std::isinf(c))
        if(c>1e7)
          sig2[i] = 2*a;
        else
          sig2[i] = 0.5*( a + c );

      } else {
        c = sig2[i];

        sig2[i] = 0.5*( a + c );

      }


      for (uint32_t j=P.col[i]; j < P.col[i+1]; j++) {
        sum_i += exp( -P.val[j] * sig2[i] );
      }

      // residual value
      fval = sum_i - lambda;

      if (sum_i == 0) sum_i = std::numeric_limits<dataPoint>::min();


    } // ... (bisection)


    // update values in P matrix
    sum_i = 0;
    for (uint32_t j=P.col[i]; j < P.col[i+1]; j++) {
      P.val[j] = exp( -P.val[j] * sig2[i] );
      sum_i += P.val[j];
    }

    // column-stochastic
    for (uint32_t j=P.col[i]; j < P.col[i+1]; j++)
      P.val[j] /= sum_i;

    // override lambda value of leaf node?
    if ( dropLeafEdge && (P.col[i+1]-P.col[i] == 1) ) P.val[ P.col[i] ] = 0;
  }

  free( sig2 );

}
template
void lambdaRescaling( sparse_matrix<float> P, float lambda, bool dist, bool dropLeafEdge );
template
void lambdaRescaling( sparse_matrix<double> P, double lambda, bool dist, bool dropLeafEdge );
