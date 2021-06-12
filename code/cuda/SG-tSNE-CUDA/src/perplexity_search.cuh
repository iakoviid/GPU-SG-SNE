
#ifndef BSR_PQ_CUH
#define BSR_PQ_CUH
#include "types.hpp"
#include <limits>
#include <iostream>

template <class dataPoint>
void equalizeVertex(dataPoint *val_P, dataPoint *distances,
                    dataPoint perplexity, int nn);
template <class dataPoint>
sparse_matrix<dataPoint> perplexityEqualization(int *I, dataPoint *D, int n,
                                                int nn, dataPoint u);
#endif
