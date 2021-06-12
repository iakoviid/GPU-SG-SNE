/*!
  \file   utils.hpp
  \brief  Auxilliary utilities.

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef UTILS_HPP
#define UTILS_HPP

#include "sparsematrix.hpp"
#include "types.hpp"
// ============================== GENERAL UTILITIES

//! Print the list of parameters
/*!

  \param P      A structure containing SG-t-SNE parameters
*/
void printParams(tsneparams P);


//! Random number generation from a normal distribution
/*!
  \return A random number drawn from a normal distribution
*/
coord randn();


//! Inline function to get the sign of the input number
/*!
*/
static inline coord sign(coord x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }

// ============================== IO UTILITIES

//! Build sparse CSC matrix from Matrix Market file
/*!

  \param filename  Name of Matrix Market file
  \return          The CSC matrix
*/
sparse_matrix buildPFromMTX( const char *filename );

//! Read high-dimensional data points from Matrix Market file.
/*!

  \param filename    Name of the file
  \param[out] n      Number of data points
  \param[out] d      Number of dimensions
  \return            The high-dimensional data [d-by-n]
*/
coord * readXfromMTX( const char *filename, int *n, int *d );

//! Save embedding in a binary file, named embedding.bin
/*!

  \param y      Embedding coordinates [d-by-n]
  \param n      Number of data points
  \param d      Number of embedding dimensions
*/
void extractEmbedding( coord *y, int n, int d );

//! Save embedding in a text file, named embedding.bin
/*!

  \param y      Embedding coordinates [d-by-n]
  \param n      Number of data points
  \param d      Number of embedding dimensions
*/
void extractEmbeddingText( coord *y, int n, int d );


//! Support of loading input data, using conventional t-SNE
/*!
*/
bool vdm_load_data(coord** data, int* n, int* d, int* no_dims, coord* theta, coord* perplexity, int* rand_seed, int* max_iter);

//! Support of saving input data, using conventional t-SNE
/*!
*/
void vdm_save_data(coord* data, int* landmarks, coord* costs, int n, int d);

#endif /* UTILS_HPP */
