/*!
  \file   sgtsne.hpp
  \brief  SG-t-SNE-Pi header with structure and function definitions.

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.
  */

#ifndef SGTSNE_CUH
#define SGTSNE_CUH

//! List of SG-t-SNE-Pi parameters
/*!
  A list of parameters available in SG-t-SNE-Pi, with default values
  specified.
*/
typedef struct {

  int    d         = 2;         //!< Number of embedding dimensions
  double lambda    = 1;         //!< λ rescaling parameter
  double alpha     = 12;        //!< Early exaggeration multiplier
  int    maxIter   = 1000;      //!< Maximum number of iterations
  int    earlyIter = 250;       //!< Number of early exaggeration iterations
  int    n         = 0;         //!< Number of vertices
  double h         = -1;        //!< Grid side length (accuracy control)
  bool   dropLeaf  = false;     //!< Drop edges originating from leaf nodes?
  int    np        = 0;         //!< Number of CILK workers (processes)

} tsneparams;



//! Embed the sparse stochastic graph P
/*!
  Compute the embedding of the input stochastic graph P.
  A list of parameters are defined in the tsneparams structure.


  \return [d-by-N] The embedding coordinates
*/
coord* sgtsneGPU(sparse_matrix P,            //!< The sparse stochastic graph P in CSC storage
              tsneparams params,          //!< A struct with the SG-t-SNE parameters
              coord *y_in = nullptr,      //!< [Optional] The embedding coordinates
              double **timeInfo = nullptr //!< [Optional] Returns timing information
              );



//! Perplexity equalization
/*!

  The input as the kNN graph of a dataset, of size [(k+1)-by-N]. The
  (k+1) input must contain the self-loop as the first element. Both
  the indices and the distances are needed.


  \return The CSC sparse column-stochastic all-kNN, after perplexity equalization.
*/
sparse_matrix perplexityEqualizationGPU( int *I,    //!< [(k+1)-by-N] array with the neighbor IDs
                                      double *D, //!< [(k+1)-by-N] array with the neighbor distances
                                      int n,     //!< [scalar] Number of data points N
                                      int nn,    //!< [scalar] Number of neighbors k
                                      double u   //!< [scalar] Perplexity u
                                      );


#endif
