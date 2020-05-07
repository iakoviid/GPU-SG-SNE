#include "utils.hpp"
#include <cmath>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>     // std::string, std::to_string

void extractEmbeddingText( coord *y, int n, int d ){

  std::ofstream f ("embedding.txt");

  if (f.is_open())
    {
      for (int i = 0 ; i < n ; i++ ){
        for (int j = 0 ; j < d ; j++ ){
          f << y[i*d + j] << " ";
        }
        f << std::endl;
      }

      f.close();
    }

}

void extractEmbedding( double *y, int n, int d ){

  FILE * pFile;
  pFile = fopen ("embedding.bin", "wb");

  fwrite(y , sizeof(double), n*d, pFile);
  fclose(pFile);
  return;
}
void printParams(tsneparams P){

  std::cout << "Number of vertices: " << P.n << std::endl
            << "Embedding dimensions: " << P.d << std::endl
            << "Rescaling parameter λ: " << P.lambda << std::endl
            << "Early exag. multiplier α: " << P.alpha << std::endl
            << "Maximum iterations: " << P.maxIter << std::endl
            << "Early exag. iterations: " << P.earlyIter << std::endl
            << "Box side length h: " << P.h << std::endl
            << "Drop edges originating from leaf nodes? " << P.dropLeaf << std::endl
            << "Number of processes: " << P.np << std::endl;

}



double tsne_stop_timer(const char * event_name, timeval begin){
  struct timeval end;
  gettimeofday(&end, NULL);
  double stime = ((double) (end.tv_sec - begin.tv_sec) * 1000 ) +
    ((double) (end.tv_usec - begin.tv_usec) / 1000 );
  stime = stime / 1000;
#ifdef PRINT_DEBUG_TIME
  printf("%-20s : %8.4lf s\n",event_name, stime);
#endif
  return(stime);
}



double randn() {
  double x, y, radius;
  do {
    x = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    y = 2 * (rand() / ((double) RAND_MAX + 1)) - 1;
    radius = (x * x) + (y * y);
  } while((radius >= 1.0) || (radius == 0.0));
  radius = sqrt(-2 * log(radius) / radius);
  x *= radius;
  y *= radius;
  return x;
}
