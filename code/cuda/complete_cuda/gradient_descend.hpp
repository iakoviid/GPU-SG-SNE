#ifndef GRADIENT_HPP
#define GRADIENT_HPP
#include "common.hpp"
#include "tsne.cuh"
#include "timers.hpp"
#include "utils.cuh"
int getBestGridSize(int nGrid);
coord computeFrepulsive_interpCPU(coord *Frep, coord *y, int n, int d, double h,
                               int np);

#endif
