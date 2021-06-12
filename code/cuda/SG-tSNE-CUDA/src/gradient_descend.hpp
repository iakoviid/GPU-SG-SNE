#ifndef GRADIENT_HPP
#define GRADIENT_HPP
#include "common.hpp"
#include "types.hpp"
#include "timers.hpp"
#include "Frep.hpp"
#include "pq.hpp"

template <class dataPoint>
coord compute_gradientCPU(dataPoint *dy, double *timeFrep, double *timeFattr,
                           tsneparams params, dataPoint *y,sparse_matrix P);

void kl_minimizationCPU(coord *y, tsneparams params, sparse_matrix P);
#endif
