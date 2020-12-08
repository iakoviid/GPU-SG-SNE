
#ifndef SGTSNE_CUH
#define SGTSNE_CUH

#include "types.hpp"
#include "common.cuh"
#include "utils.cuh"
#include "utils_cuda.cuh"
#include "sparsematrix.cuh"
#include "sparsematrix.hpp"
#include "gradient_descend.cuh"
//! Embed the sparse stochastic graph P
/*!
  Compute the embedding of the input stochastic graph P.
  A list of parameters are defined in the tsneparams structure.


  \return [d-by-N] The embedding coordinates
*/

coord *sgtsneCUDA(
    sparse_matrix P,       //!< The sparse stochastic graph P in CSR storage
    tsneparams params,     //!< A struct with the SG-t-SNE parameters
    coord *y_in = nullptr, //!< [Optional] The embedding coordinates
    double **timeInfo = nullptr //!< [Optional] Returns timing information
);


#endif /* _SGTSNE_CUH_ */
