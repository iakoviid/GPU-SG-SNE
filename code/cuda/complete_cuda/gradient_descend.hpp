#ifndef GRADIENT_HPP
#define GRADIENT_HPP
#include "common.hpp"
#include "tsne.cuh"
#include "timers.hpp"
#include "utils.cuh"
coord computeFrepulsive_interp(coord *Frep, coord *y, int n, int d, double h,
                               int np);

#endif
