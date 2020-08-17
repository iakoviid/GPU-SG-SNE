#ifndef GRADIENT_HPP
#define GRADIENT_HPP
#include "common.hpp"
#include "tsne.cuh"
#include "timers.hpp"
#include "utils.cuh"
#include "Frep.hpp"
#include "../csb/csb_wrapper.hpp"

template <class dataPoint>
double compute_gradientCPU(dataPoint *dy, double *timeFrep, double *timeFattr,tsneparams params, dataPoint *y);
void kl_minimizationCPU(coord *y, tsneparams params);

#endif
