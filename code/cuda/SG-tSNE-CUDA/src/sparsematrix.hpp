/*!
  \file   sparsematrix.hpp
  \brief  Basic sparse matrix routines.

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef SPARSEMATRIX_HPP
#define SPARSEMATRIX_HPP

#include "types.hpp"
//#include "utils.hpp"
#include <cstdlib>

//! Free allocated memory for CSC sparse matrix storage
/*!

  \param P      Sparse matrix in CSC format.
*/
template<typename dataPoint>
void free_sparse_matrix(sparse_matrix<dataPoint> * P);

//! Transform input matrix to stochastic
/*!

  \param P      Sparse matrix in CSC format.
  \return       Number of nodes already stochastic
*/
template<typename dataPoint>
uint32_t makeStochastic(sparse_matrix<dataPoint> P);

//! Print sparse matrix P (only print size if too large)
/*!

  \param P      Sparse matrix in CSC format.
*/
template<typename dataPoint>
void printSparseMatrix( sparse_matrix<dataPoint> P );

//! Symmetrize matrix P
/*!

  \param[in,out] P      Sparse matrix in CSC format.
*/
template<typename dataPoint>
void symmetrizeMatrix( sparse_matrix<dataPoint> *P );


//! Permute matrix P
/*!

  \param[in,out] P      Sparse matrix in CSC format.
  \param perm           Permutation vector
  \param iperm          Inverse permutation vector
*/
template<typename dataPoint>
void permuteMatrix( sparse_matrix<dataPoint> *P, int *perm, int *iperm );

#endif /* SPARSEMATRIX_HPP */
