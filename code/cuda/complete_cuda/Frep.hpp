#ifndef FREP_CPP
#define FREP_CPP

#include "common.hpp"
#include "nuconv.hpp"
#include "relocateData.hpp"
#include "utils.cuh"
#include <cmath>
#include <limits>
coord computeFrepulsive_interpCPU(coord *Frep, coord *y, int n, int d, double h,
                                  int np);
template <typename dataval>
dataval zetaAndForceCPU(dataval *const F,            // Forces
                        const dataval *const Y,      // Coordinates
                        const dataval *const Phi,    // Values
                        const uint32_t *const iPerm, // Permutation
                        const uint32_t nPts,         // #points
                        const uint32_t nDim);
#endif
