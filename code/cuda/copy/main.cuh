
#include "common.cuh"
coord* loadData(uint32_t* n,uint32_t* d,char* mode,coord** y_d,uint32_t* perplexity);
void tsneRun(coord* y_d,coord* x_d,uint32_t n,uint32_t d,uint32_t Dim,uint32_t perplexity);
void tsneRunCpu(coord* y,coord* x,uint N,uint d,uint32_t Dim,uint32_t perplexity);
coord *generateRandomCoord(uint32_t n, uint32_t d,coord scale);
void extractEmbeddingText( coord *y, int n, int d );
