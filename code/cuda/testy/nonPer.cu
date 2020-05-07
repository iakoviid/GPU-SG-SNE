#include <stdio.h>
#include <stdlib.h>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include "matrix_indexing.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <numeric>
#include <math.h>
#include <limits>
#include <thrust/device_vector.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>
#include "common.h"


double const pi = 4 * std::atan(1);


void conv1dnopad( double * const PhiGrid,
                  const double * const VGrid,
                  const double h,
                  uint32_t * const nGridDims,
                  const uint32_t nVec,
                  const uint32_t nDim,
                  const uint32_t nProc)
{

  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
    fftw_complex *K, *X, *w;
  std::complex<double> *Kc, *Xc, *wc;
  fftw_plan planc_kernel, planc_rhs, planc_inverse;

  // get h^2
  double hsq = h*h;

  // total number of grid points (VGrid's leading dimension)
  uint32_t n1=nGridDims[0];

  // FFTW plan options
  int rank = 1;
  int n[] = {static_cast<int>(n1)};
  int howmany = nVec;
  int idist = n1;
  int odist = n1;
  int istride = 1;
  int ostride = 1;
  int *inembed = NULL, *onembed = NULL;


  // allocate memory for kernel and RHS FFTs
  K = (fftw_complex *) fftw_malloc( n1 * sizeof(fftw_complex) );
  X = (fftw_complex *) fftw_malloc( n1 * nVec * sizeof(fftw_complex) );
  w = (fftw_complex *) fftw_malloc( n1 * sizeof(fftw_complex) );

  Kc = reinterpret_cast<std::complex<double> *> (K);
  Xc = reinterpret_cast<std::complex<double> *> (X);
  wc = reinterpret_cast<std::complex<double> *> (w);

  // get twiddle factors
  for (uint32_t i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );

  for(int i=0;i<n1;i++){
      Kc[i] = 0.0;
      for(int j=0;j<nVec;j++){
        Xc[i*nVec+j]=0.0;
      }
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM



  // ~~~~~~~~~~~~~~~~~~~~ SETUP FFTW PLANS

  planc_kernel = fftw_plan_dft_1d(n1, K, K, FFTW_FORWARD, FFTW_ESTIMATE);

  planc_rhs = fftw_plan_many_dft(rank, n, howmany, X, inembed,
                                 istride, idist,
                                 X, onembed,
                                 ostride, odist,
                                 FFTW_FORWARD, FFTW_ESTIMATE);

  planc_inverse = fftw_plan_many_dft(rank, n, howmany, X, inembed,
                                     istride, idist,
                                     X, onembed,
                                     ostride, odist,
                                     FFTW_BACKWARD, FFTW_ESTIMATE);

  // ============================== EVEN FREQUENCIES

  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL

  for (uint32_t i=0; i<n1; i++) {
    std::complex<double> tmp( kernel1d( hsq, i ), 0 );
             Kc[i]    += tmp;
    if (i>0) Kc[n1-i] += tmp;
  }


  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t i=0; i<n1; i++) {
      Xc[ SUB2IND2D(i, iVec ,n1) ] =
        VGrid[ SUB2IND2D(i, iVec, n1) ];
    }
  }
  //printf("======================Preefft====Host==========================\n" );
  for(int i=0;i<n1*nVec;i++){
    //printf("Xc[%d]=%lf+%lf i\n",i,Xc[i].real(),Xc[i].imag() );
  }

  // ---------- execute kernel plan
  fftw_execute(planc_kernel);


  // ---------- execute RHS plan
  fftw_execute(planc_rhs);

  //printf("===============================HOST======================================\n" );
  for(int i=0;i<n1;i++){
  //  printf("Host Kc[%d]=%lf+%lf i\n",i,Kc[i].real(),Kc[i].imag() );
  }
  //printf("===============================HOST======================================\n" );
//  printf("=======================PostFFT===Host==========================\n" );
  for(int i=0;i<n1*nVec;i++){
  //  printf("Xc[%d]=%lf+%lf i\n",i,Xc[i].real(),Xc[i].imag() );
  }
  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (uint32_t jVec=0; jVec<nVec; jVec++) {
    for (uint32_t i=0; i<n1; i++){
      Xc[SUB2IND2D(i,jVec,n1)] = Xc[SUB2IND2D(i,jVec,n1)] *
        Kc[i];
    }
  }
  //printf("=======================PostProduxt===Host==========================\n" );

  for(int i=0;i<n1*nVec;i++){
    //printf("Xc[%d]=%lf+%lf i\n",i,Xc[i].real(),Xc[i].imag() );
  }

  // ---------- execute inverse plan
  fftw_execute(planc_inverse);

  // ---------- (no conjugate multiplication)

  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t i=0; i<n1; i++){
      PhiGrid[ SUB2IND2D(i, iVec, n1) ] =
        Xc[ SUB2IND2D(i, iVec, n1) ].real();
    }
  }

  // ============================== ODD FREQUENCIES

  for(int i=0;i<n1;i++){
      Kc[i] = 0.0;
      for(int j=0;j<nVec;j++){
        Xc[i*nVec+j]=0.0;
      }
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (uint32_t i=0; i<n1; i++) {
    std::complex<double> tmp( kernel1d( hsq, i ), 0 );
             Kc[i]    += tmp;
    if (i>0) Kc[n1-i] -= tmp;
  }

  for (uint32_t i=0; i<n1; i++) {
    Kc[i] *= wc[i];
  }

  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS

  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t i=0; i<n1; i++) {
      Xc[ SUB2IND2D(i, iVec ,n1) ] =
        VGrid[ SUB2IND2D(i, iVec, n1) ] * wc[i];
    }
  }

  // ---------- execute kernel plan
  fftw_execute(planc_kernel);

  // ---------- execute RHS plan
  fftw_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (uint32_t jVec=0; jVec<nVec; jVec++) {
    for (uint32_t i=0; i<n1; i++){
      Xc[SUB2IND2D(i,jVec,n1)] = Xc[SUB2IND2D(i,jVec,n1)] *
        Kc[i];
    }
  }

  // ---------- execute inverse plan
  fftw_execute(planc_inverse);


  // ---------- data normalization
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t i=0; i<n1; i++) {
      Xc[ SUB2IND2D(i, iVec, n1) ] =
        Xc[ SUB2IND2D(i, iVec, n1) ] *
        std::conj(wc[i]);
    }
  }

  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t i=0; i<n1; i++){
      PhiGrid[ SUB2IND2D(i, iVec, n1) ] +=
        Xc[ SUB2IND2D(i, iVec, n1) ].real();
    }
  }
  printf("-----------------------Host End---------------------------------\n" );
  for(int i=0;i<nVec*n1;i++){
      PhiGrid[i] *= (0.5 / n1);
      printf("Phi[%d]=%lf\n",i,PhiGrid[i] );
  }



  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftw_destroy_plan( planc_kernel );
  fftw_destroy_plan( planc_rhs );
  fftw_destroy_plan( planc_inverse );


  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftw_free( K );
  fftw_free( X );
  fftw_free( w );

}
