/*!
  \file   types.hpp
  \brief  Type definitions.

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef TYPES_HPP
#define TYPES_HPP

#include <stdint.h>

typedef double coord;   //!< Data-type of embedding coordinates
typedef double matval;   //!< Data-type of sparse matrix elements
typedef int matidx; //!< Data-type of sparse matrix indices


//! Sparse matrix structure
/*!
  Custom structure to hold the elements of a sparse matrix format.
*/
typedef struct {

  int m;   //!< Number of rows
  int n;   //!< Number of columns
  int nnz; //!< Number of nonzero elements
  int format;//format of the matrix

  matidx *row; //!< Rows offset (N+1 length)
  matidx *col; //!< Columns indices (NNZ length)
  matval *val; //!< Values (NNZ length)


} sparse_matrix;


//! List of SG-t-SNE-Pi parameters
/*!
  A list of parameters available in SG-t-SNE-Pi, with default values
  specified.
*/
typedef struct {

  int d = 2;             //!< Number of embedding dimensions
  double lambda = 1;     //!< λ rescaling parameter
  double alpha = 12;     //!< Early exaggeration multiplier
  int maxIter = 1000;    //!< Maximum number of iterations
  int earlyIter = 250;   //!< Number of early exaggeration iterations
  int n = 0;             //!< Number of vertices
  double h = -1;         //!< Grid side length (accuracy control)
  bool dropLeaf = false; //!< Drop edges originating from leaf nodes?
  int np = 0;            //!< Number of CILK workers (processes)

} tsneparams;

#endif /* TYPES_HPP */
