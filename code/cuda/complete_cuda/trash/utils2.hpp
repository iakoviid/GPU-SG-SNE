/*!
  \file   utils.hpp
  \brief  Auxilliary utilities.

  \author Dimitris Floros
  \date   2019-06-20
*/


#ifndef UTILS_HPP
#define UTILS_HPP
#include "common.cpp"
#include "tsne.cuh"


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
double randn();


//! Inline function to get the sign of the input number
/*!
*/
static inline double sign(double x) { return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0)); }


//! Save embedding in a binary file, named embedding.bin
/*!

  \param y      Embedding coordinates [d-by-n]
  \param n      Number of data points
  \param d      Number of embedding dimensions
*/
void extractEmbedding( double *y, int n, int d );

//! Save embedding in a text file, named embedding.bin
/*!

  \param y      Embedding coordinates [d-by-n]
  \param n      Number of data points
  \param d      Number of embedding dimensions
*/
void extractEmbeddingText( double *y, int n, int d );
struct timeval tsne_start_timer(){
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv;
}


double tsne_stop_timer(const char * event_name, timeval begin);
#endif /* UTILS_HPP */
