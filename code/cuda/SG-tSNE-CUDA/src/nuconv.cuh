#ifndef NUCONV_CUH
#define NUCONV_CUH
#include "common.cuh"
#include "gridding1.cuh"
#include "non_periodic_convF.cuh"
#include "non_periodic_convD.cuh"
#include <thrust/device_vector.h>
#define E_LIMITF 1.19209290e-007
#define E_LIMITD 2.22045e-16
template <class dataPoint,class Complext>
void nuconv(dataPoint *PhiScat, dataPoint *y, dataPoint *VScat, int *ib, int n,
            int d, int m, int nGridDim, double *timeInfo, cufftHandle &plan,
            cufftHandle &plan_rhs, dataPoint *VGrid,
            dataPoint *PhiGrid, Complext *Kc,
            Complext *Xc);
#endif
