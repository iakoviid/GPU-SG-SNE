/*!
  \file   graph_rescaling.hpp
  \brief  Routines implementing lambda-based graph rescaling

*/



#ifndef GRAPH_RESCALING_CUH
#define GRAPH_RESCALING_CUH

#include "types.hpp"
#include "utils_gpu.cuh"
#include <thrust/device_vector.h>
#include "common.cuh"
//! Rescale given column-stochastic graph, using specified lambda parameter
/*!
*/
void lambdaRescalingGPU( sparse_matrix P,        //!< Column-stocastic CSC matrix
                      double lambda,          //!< λ rescaling parameter
                      bool dist=false,        //!< [optional] Consider input as distance?
                      bool dropLeafEdge=false //!< [optional] Remove edges from leaf nodes?
                      );
#endif /* GRAPH_RESCALING_HPP */
