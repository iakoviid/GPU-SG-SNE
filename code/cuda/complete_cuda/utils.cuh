#ifndef UTILS_HPP
#define UTILS_HPP
#include "common.cuh"
#include "tsne.cuh"
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include <string> // std::string, std::to_string
void extractEmbeddingText(coord *y, int n, int d);

void extractEmbedding(double *y, int n, int d);
void printParams(tsneparams P);

__device__ __host__ static inline double sign(double x) {

  return (x == .0 ? .0 : (x < .0 ? -1.0 : 1.0));
}

template <class dataPoint>
__global__ void addScalar(dataPoint *a, dataPoint scalar, uint32_t length);

template <typename T>
__global__ void initKernel(T *devPtr, const T val, const size_t nwords);
#endif
