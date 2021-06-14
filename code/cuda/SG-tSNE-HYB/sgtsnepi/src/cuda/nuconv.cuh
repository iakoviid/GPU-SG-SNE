#ifndef NUCONV_CUH
#define NUCONV_CUH
#include "common.cuh"
#include "gridding.cuh"
#include "non_periodic_convF.cuh"
#include "non_periodic_convD.cuh"
#include <thrust/device_vector.h>
template <class dataPoint,class Complext>
void nuconv(dataPoint *PhiScat, dataPoint *y, dataPoint *VScat,  int n,
            int d, int m, int nGridDim, double *timeInfo, cufftHandle &plan,
            cufftHandle &plan_rhs, dataPoint *VGrid,
            dataPoint *PhiGrid, Complext *Kc,
            Complext *Xc);
#endif
