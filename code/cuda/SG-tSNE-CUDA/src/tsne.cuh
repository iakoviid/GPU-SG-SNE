/*!
  \file   sgtsne.hpp
  \brief  SG-t-SNE-Pi header with structure and function definitions.

  The main procedure definition, responsible for parsing the data
  and the parameters, preprocessing the input, running the
  gradient descent iterations and returning.

  \author Dimitris Floros
  \date   2019-06-21
*/


#ifndef SGTSNE_HPP
#define SGTSNE_HPP
#include "common.hpp"


//! Sparse matrix structure in CSC format
/*!
  Custom structure to hold the elements of a CSC sparse matrix format.

typedef struct {

  int m;        //!< Number of rows
  int n;        //!< Number of columns
  int nnz;      //!< Number of nonzero elements

  matidx * row; //!< Rows indices (NNZ length)
  matidx * col; //!< Columns offset (N+1 length)
  matval * val; //!< Values (NNZ length)

} sparse_matrix;
*/
// include utility functions
//#include "utils.hpp"

void kl_minimizationCPU(coord* Y,  //!< Embedding coordinates (output)
                     tsneparams param, //!< t-SNE parameters
                     double **timeInfo //!< [Optional] Timing information (output)
                     );
void kl_minimization(coord *y, tsneparams params);
//void kl_minimization(coord* y,tsneparams params);

#endif /* _SGTSNE_H_ */
