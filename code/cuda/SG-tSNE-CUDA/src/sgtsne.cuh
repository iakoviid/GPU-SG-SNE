
#ifndef SGTSNE_CUH
#define SGTSNE_CUH

#include "types.hpp"
#include "common.cuh"
#include "utils.cuh"
#include "utils_cuda.cuh"
#include <sys/time.h>

#include "gradient_descend.cuh"

//! Embed the sparse stochastic graph P
/*!
  Compute the embedding of the input stochastic graph P.
  A list of parameters are defined in the tsneparams structure.


  \return [d-by-N] The embedding coordinates
*/
template <class dataPoint>
dataPoint *sgtsneCUDA(
    sparse_matrix<dataPoint> *P,       //!< The sparse stochastic graph P in CSR storage
    tsneparams params,     //!< A struct with the SG-t-SNE parameters
    dataPoint *y_in , //!< [Optional] The embedding coordinates
    double *timeInfo  //!< [Optional] Returns timing information
);


#endif /* _SGTSNE_CUH_ */
