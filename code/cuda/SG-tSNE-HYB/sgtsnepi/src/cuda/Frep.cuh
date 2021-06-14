/*!
  \file   Frep.cuh
  \brief  Implementation for the apprpoximation of the Repulsive term header.

  \author Iakovidis Ioannis
  \date   2021-06-14
*/
#ifndef FREP_CUH
#define FREP_CUH
#include "common.cuh"
#include "nuconv.cuh"

#include "utils_cuda.cuh"

#include <thrust/device_vector.h>
#include <thrust/functional.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/reduce.h>
#include <thrust/sequence.h>

double computeFrepulsive_gpu(double *Freph, double *yh, int n, int d, double h,
                             double *timeInfo);
coord computeFrepulsive_GPU(coord *Freph, coord *yh, int n, int d, coord h,
                            double *timeInfo);
#endif
