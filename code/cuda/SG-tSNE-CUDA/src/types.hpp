/*!
  // ----- timing
  // ----- timing
  // ----- timing
  \file   types.hpp
  \brief  Type definitions.

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef TYPES_HPP
#define TYPES_HPP

#include <stdint.h>

typedef float coord;   //!< Data-type of embedding coordinates
typedef float matval;   //!< Data-type of sparse matrix elements
typedef int matidx; //!< Data-type of sparse matrix indices


//! Sparse matrix structure
/*!
  Custom structure to hold the elements of a sparse matrix format.
*/
template<typename dataPoint>
struct sparse_matrix {

  int m;   //!< Number of rows
  int n;   //!< Number of columns
  int nnz; //!< Number of nonzero elements
  int format;//format of the matrix

  matidx *row; //!< Rows offset (N+1 length)
  matidx *col; //!< Columns indices (NNZ length)
  dataPoint *val; //!< Values (NNZ length)

  int nnzb;
  int blockSize;
  int blockRows;

  unsigned int* ell_cols,* coo_col_ids,*coo_row_ids;
  dataPoint* ell_data, *coo_data;
  int coo_size, elements_in_rows;

};


//! List of SG-t-SNE-Pi parameters
/*!
  A list of parameters available in SG-t-SNE-Pi, with default values
  specified.
*/
typedef struct {

  int d = 2;             //!< Number of embedding dimensions
  float lambda = 1;     //!< λ rescaling parameter
  float alpha = 12;     //!< Early exaggeration multiplier
  int maxIter = 1000;    //!< Maximum number of iterations
  int earlyIter = 250;   //!< Number of early exaggeration iterations
  int n = 0;             //!< Number of vertices
  float h = -1;         //!< Grid side length (accuracy control)
  bool dropLeaf = false; //!< Drop edges originating from leaf nodes?
  int np = 0;            //!< Number of CILK workers (processes)
  int format=2;
  int bs = 0;
  char* method="metis";
  int ComputeError=-1;
  int ng=50;
  int sim=0;

} tsneparams;

#endif /* TYPES_HPP */
