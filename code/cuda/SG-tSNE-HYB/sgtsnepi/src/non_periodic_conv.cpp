
#include <iostream>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include <cilk/cilk.h>
#include "matrix_indexing.hpp"


double kernel1d(double hsq, double i) {
  return pow(1.0 + hsq * i*i, -2);
}

double kernel2d(double hsq, double i, double j) {
  return pow(1.0 + hsq * ( i*i + j*j ), -2);
}

double kernel3d(double hsq, double i, double j, double k) {
  return pow(1.0 + hsq * ( i*i + j*j + k*k ), -2);
}

double const pi = 4 * std::atan(1);

#include "convolution_nopadding_helper.cpp"

void conv1dnopad( double * const PhiGrid,
                  const double * const VGrid,
                  const double h,
                  uint32_t * const nGridDims,
                  const uint32_t nVec,
                  const uint32_t nDim,
                  const uint32_t nProc) {

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
  cilk_for (uint32_t i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );

  Kc[0:(n1)] = 0.0;
  Xc[0:(n1*nVec)] = 0.0;
  

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM

  fftw_init_threads();
  fftw_plan_with_nthreads(nProc);
  
  
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

  // ---------- (no conjugate multiplication)
  
  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t i=0; i<n1; i++){
      PhiGrid[ SUB2IND2D(i, iVec, n1) ] =
        Xc[ SUB2IND2D(i, iVec, n1) ].real();
    }
  }

  // ============================== ODD FREQUENCIES

  Kc[0:n1] = 0.0; 
  Xc[0:n1*nVec] = 0.0;
  
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

  PhiGrid[ 0:n1*nVec ] *= (0.5 / n1);

  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftw_destroy_plan( planc_kernel );
  fftw_destroy_plan( planc_rhs );
  fftw_destroy_plan( planc_inverse );

  fftw_cleanup_threads();
  
  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftw_free( K );
  fftw_free( X );
  fftw_free( w );
  
}


void conv2dnopad( double * const PhiGrid,
                  const double * const VGrid,
                  const double h,
                  uint32_t * const nGridDims,
                  const uint32_t nVec,
                  const uint32_t nDim,
                  const uint32_t nProc ) {
  
  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  fftw_complex *K, *X, *w;
  std::complex<double> *Kc, *Xc, *wc;
  fftw_plan planc_kernel, planc_rhs, planc_inverse;
  
  // get h^2
  double hsq = h*h;

  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1=nGridDims[0];
  uint32_t n2=nGridDims[1];

  int rank = 2;
  int n[] = {static_cast<int>(n1), static_cast<int>(n2)};
  int howmany = nVec;
  int idist = n1*n2;
  int odist = n1*n2;
  int istride = 1;
  int ostride = 1;
  int *inembed = NULL, *onembed = NULL;

  // allocate memory for kernel and RHS FFTs
  K = (fftw_complex *) fftw_malloc( n1 * n2 * sizeof(fftw_complex) );
  X = (fftw_complex *) fftw_malloc( n1 * n2 * nVec * sizeof(fftw_complex) );
  w = (fftw_complex *) fftw_malloc( n1 * sizeof(fftw_complex) );

  Kc = reinterpret_cast<std::complex<double> *> (K);
  Xc = reinterpret_cast<std::complex<double> *> (X);
  wc = reinterpret_cast<std::complex<double> *> (w);

  // get twiddle factors
  cilk_for (uint32_t i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );
  
  Kc[0:(n1*n2)] = 0.0;
  Xc[0:(n1*n2*nVec)] = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM

  fftw_init_threads();
  fftw_plan_with_nthreads(nProc);
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP FFTW PLANS

  planc_kernel = fftw_plan_dft_2d(n1, n2, K, K, FFTW_FORWARD, FFTW_ESTIMATE);

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

  // ============================== EVEN-EVEN
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      std::complex<double> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      = tmp;
      if (i>0) Kc[SUB2IND2D(i,j,n1)] += kernel2d( hsq,n1- i, j ) ;
      if (j>0) Kc[SUB2IND2D(i,j,n1)] += kernel2d( hsq,i,n2- j );
      if (i>0 && j>0) Kc[SUB2IND2D(i,j,n1)] += kernel2d( hsq,n1- i,n2- j );
    }
  }
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ];
      }
    }
  }


  // ---------- execute kernel plan
  fftw_execute(planc_kernel);
  
  // ---------- execute RHS plan
  fftw_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (uint32_t jVec=0; jVec<nVec; jVec++) {
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftw_execute(planc_inverse);

  // ---------- (no conjugate multiplication)

  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] =
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }

  // ============================== ODD-EVEN

  Kc[0:n1*n2] = 0.0; 
  Xc[0:n1*n2*nVec] = 0.0;
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
  
  for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      std::complex<double> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      = tmp;
      if (i>0) Kc[SUB2IND2D(i,j,n1)] -= kernel2d( hsq,n1- i, j );
      if (j>0) Kc[SUB2IND2D(i,j,n1)] += kernel2d( hsq,i,n2- j );
      if (i>0 && j>0) Kc[SUB2IND2D(i,j,n1)] -= kernel2d( hsq,n1- i,n2- j );
    }
  }



  for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      Kc[SUB2IND2D(i,j,n1)] *= wc[i];
    }
  }
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] * wc[i];
      }
    }
  }

    
  // ---------- execute kernel plan
  fftw_execute(planc_kernel);
  
  // ---------- execute RHS plan
  fftw_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (uint32_t jVec=0; jVec<nVec; jVec++) {
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftw_execute(planc_inverse);

  // ---------- data normalization
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec, n1,n2) ] =
          Xc[ SUB2IND3D(i, j,iVec, n1,n2) ] *
          std::conj(wc[i]);
      }
    }
  }

  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] +=
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }


  // ============================== EVEN-ODD

  Kc[0:n1*n2] = 0.0; 
  Xc[0:n1*n2*nVec] = 0.0;
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
    for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      std::complex<double> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      = tmp;
      if (i>0) Kc[SUB2IND2D(i,j,n1)] += kernel2d( hsq,n1- i, j );
      if (j>0) Kc[SUB2IND2D(i,j,n1)] -= kernel2d( hsq,i,n2- j );
      if (i>0 && j>0) Kc[SUB2IND2D(i,j,n1)] -= kernel2d( hsq,n1- i,n2- j );
    }
  }


  
  for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      Kc[SUB2IND2D(i,j,n1)] *= wc[j];
    }
  }
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] * wc[j];
      }
    }
  }

    
  // ---------- execute kernel plan
  fftw_execute(planc_kernel);
  
  // ---------- execute RHS plan
  fftw_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (uint32_t jVec=0; jVec<nVec; jVec++) {
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftw_execute(planc_inverse);

  // ---------- data normalization
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec, n1,n2) ] =
          Xc[ SUB2IND3D(i, j,iVec, n1,n2) ] *
          std::conj(wc[j]);
      }
    }
  }

  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] +=
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }


  // ============================== ODD-ODD

  Kc[0:n1*n2] = 0.0; 
  Xc[0:n1*n2*nVec] = 0.0;
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP KERNEL
    for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      std::complex<double> tmp( kernel2d( hsq, i, j ), 0 );
      Kc[SUB2IND2D(i,j,n1)]      = tmp;
      if (i>0) Kc[SUB2IND2D(i,j,n1)] -=  kernel2d( hsq,n1- i, j );
      if (j>0) Kc[SUB2IND2D(i,j,n1)] -= kernel2d( hsq,i,n2- j );
      if (i>0 && j>0) Kc[SUB2IND2D(i,j,n1)] += kernel2d( hsq,n1- i,n2- j );
    }
  }


  for (uint32_t j=0; j<n2; j++) {
    for (uint32_t i=0; i<n1; i++) {
      Kc[SUB2IND2D(i,j,n1)] *= wc[j]*wc[i];
    }
  }
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP RHS
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec ,n1, n2) ] =
          VGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] * wc[j] * wc[i];
      }
    }
  }

    
  // ---------- execute kernel plan
  fftw_execute(planc_kernel);
  
  // ---------- execute RHS plan
  fftw_execute(planc_rhs);

  // ~~~~~~~~~~~~~~~~~~~~ HADAMARD PRODUCT
  for (uint32_t jVec=0; jVec<nVec; jVec++) {
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        Xc[SUB2IND3D(i,j,jVec,n1,n2)] = Xc[SUB2IND3D(i,j,jVec,n1,n2)] *
          Kc[SUB2IND2D(i,j,n1)];
      }
    }
  }

  // ---------- execute plan
  fftw_execute(planc_inverse);

  // ---------- data normalization
  for (uint32_t iVec=0; iVec<nVec; iVec++) {
    for (uint32_t j=0; j<n2; j++) {
      for (uint32_t i=0; i<n1; i++) {
        Xc[ SUB2IND3D(i, j, iVec, n1,n2) ] =
          Xc[ SUB2IND3D(i, j,iVec, n1,n2) ] *
          std::conj(wc[i]) * std::conj(wc[j]);
      }
    }
  }

  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] +=
          Xc[ SUB2IND3D(i, j, iVec, n1, n2) ].real();
      }
    }
  }
  
  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t j=0; j<n2; j++){
      for (uint32_t i=0; i<n1; i++){
        PhiGrid[ SUB2IND3D(i, j, iVec, n1, n2) ] *= 0.25 / ((double) n1*n2);
      }
    }
  }

  
  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftw_destroy_plan( planc_kernel );
  fftw_destroy_plan( planc_rhs );
  fftw_destroy_plan( planc_inverse );

  fftw_cleanup_threads();

  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftw_free( K );
  fftw_free( X );
  fftw_free( w );
  
}

void conv3dnopad( double * const PhiGrid,
                  const double * const VGrid,
                  const double h,
                  uint32_t * const nGridDims,
                  const uint32_t nVec,
                  const uint32_t nDim,
                  const uint32_t nProc ) {
  
  // ~~~~~~~~~~~~~~~~~~~~ DEFINE VARIABLES
  fftw_complex *K, *X, *w;
  std::complex<double> *Kc, *Xc, *wc;
  fftw_plan planc_kernel, planc_rhs, planc_inverse;
  
  // get h^2
  double hsq = h*h;
  
  // find the size of the last dimension in FFTW (add padding)
  uint32_t n1=nGridDims[0];
  uint32_t n2=nGridDims[1];
  uint32_t n3=nGridDims[2];

  int rank = 3;
  int n[] = {static_cast<int>(n1), static_cast<int>(n2), static_cast<int>(n3)};
  int howmany = nVec;
  int idist = n1*n2*n3;
  int odist = n1*n2*n3;
  int istride = 1;
  int ostride = 1;
  int *inembed = NULL, *onembed = NULL;

  // allocate memory for kernel and RHS FFTs
  K = (fftw_complex *) fftw_malloc( n1 * n2 * n3 * sizeof(fftw_complex) );
  X = (fftw_complex *) fftw_malloc( n1 * n2 * n3 * nVec * sizeof(fftw_complex) );
  w = (fftw_complex *) fftw_malloc( n1 * sizeof(fftw_complex) );

  Kc = reinterpret_cast<std::complex<double> *> (K);
  Xc = reinterpret_cast<std::complex<double> *> (X);
  wc = reinterpret_cast<std::complex<double> *> (w);

  // get twiddle factors
  cilk_for (uint32_t i=0; i<nGridDims[0]; i++)
    wc[i] = std::polar(1.0, -2*pi*i/(2*nGridDims[0]) );
  
  Kc[0:(n1*n2*n3)] = 0.0;
  Xc[0:(n1*n2*n3*nVec)] = 0.0;

  // ~~~~~~~~~~~~~~~~~~~~ SETUP PARALLELISM

  fftw_init_threads();
  fftw_plan_with_nthreads(nProc);
  
  // ~~~~~~~~~~~~~~~~~~~~ SETUP FFTW PLANS

  planc_kernel = fftw_plan_dft_3d(n1, n2, n3, K, K, FFTW_FORWARD, FFTW_ESTIMATE);

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

  // ============================== 8 KERNELS
  
  eee( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  oee( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  eoe( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  ooe( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  eeo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  oeo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  eoo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );

  ooo( PhiGrid, VGrid, Xc, Kc, wc,
       planc_kernel, planc_rhs, planc_inverse,
       n1, n2, n3, nVec, hsq );


  for (uint32_t iVec=0; iVec<nVec; iVec++){
    for (uint32_t k=0; k<n3; k++){
      for (uint32_t j=0; j<n2; j++){
        for (uint32_t i=0; i<n1; i++){
          PhiGrid[ SUB2IND4D(i, j, k, iVec, n1, n2, n3) ] *= 0.125 / ((double) n1*n2*n3);
        }
      }
    }
  }
  

  // ~~~~~~~~~~~~~~~~~~~~ DESTROY FFTW PLANS
  fftw_destroy_plan( planc_kernel );
  fftw_destroy_plan( planc_rhs );
  fftw_destroy_plan( planc_inverse );
  fftw_cleanup_threads();
  
  // ~~~~~~~~~~~~~~~~~~~~ DE-ALLOCATE MEMORIES
  fftw_free( K );
  fftw_free( X );
  fftw_free( w );

}
