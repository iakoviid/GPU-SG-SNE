#include "utils.cuh"

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
