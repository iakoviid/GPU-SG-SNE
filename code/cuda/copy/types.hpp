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
typedef uint32_t matidx; //!< Data-type of sparse matrix indices


//! Sparse matrix structure in CSC format
/*!
  Custom structure to hold the elements of a CSC sparse matrix format.
*/
typedef struct {

  int m;        //!< Number of rows
  int n;        //!< Number of columns
  int nnz;      //!< Number of nonzero elements

  matidx * row; //!< Rows indices (NNZ length)
  matidx * col; //!< Columns offset (N+1 length)
  matval * val; //!< Values (NNZ length)

} sparse_matrix;

#endif /* TYPES_HPP */
