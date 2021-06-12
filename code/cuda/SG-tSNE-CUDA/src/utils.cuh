#ifndef UTILS_CUH
#define UTILS_CUH
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdio.h>
#include "types.hpp"
#include <string> // std::string, std::to_string
#include <common.cuh>
template <class dataPoint>
void appendProgressGPU(dataPoint *yd,int n,int d,const char* name);
template<class dataPoint>
dataPoint* load10x(int n,int d);
template <class dataPoint>
void extractEmbeddingText( dataPoint *y, int n, int d,const char* name );

template <class dataPoint>
void extractEmbeddingTextT( dataPoint *y, int n, int d,const char* name );

template <class dataPoint>
void extractEmbedding( dataPoint *y, int n, int d );

void printParams(tsneparams P);
template <class dataPoint>
void savePtxt(int* csrrow,int* csrcol,dataPoint* csrval,int n,const char* name);

void ExtractTimeInfo(double* timeInfo,int n,int d,const char* fname);
template <class dataPoint>
dataPoint randn();
int getBestGridSize(int nGrid);

//! Read high-dimensional data points from Matrix Market file.
/*!

  \param filename    Name of the file
  \param[out] n      Number of data points
  \param[out] d      Number of dimensions
  \return            The high-dimensional data [d-by-n]
*/
template <class dataPoint>
dataPoint * readXfromMTX( const char *filename, int *n, int *d );
template <class dataPoint> dataPoint *generateRandomGaussianCoord(int n, int d);
template <class dataPoint> dataPoint *generateRandomCoord(int n, int d);
template <class dataPoint>
sparse_matrix<dataPoint> buildPFromMTX( const char *filename );
template <class dataPoint>
dataPoint* loadSimulationData(int n,int dim);
template <class dataPoint>
dataPoint* loadSimulationData(int n,int dim,int n_clusters);
#endif
