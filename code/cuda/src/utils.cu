#include "utils.cuh"
#define N_GRID_SIZE 137

void extractEmbeddingText( coord *y, int n, int d,const char* name ){

  std::ofstream f (name);

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
void extractEmbeddingTextT( coord *y, int n, int d,const char* name ){

  std::ofstream f (name);

  if (f.is_open())
    {
      for (int i = 0 ; i < n ; i++ ){
        for (int j = 0 ; j < d ; j++ ){
          f << y[i + j*n] << " ";
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
int getBestGridSize(int nGrid) {

  // list of FFT sizes that work "fast" with FFTW
  int listGridSize[N_GRID_SIZE] = {
      8,   9,   10,  11,  12,  13,  14,  15,  16,  20,  25,  26,  28,  32,
      33,  35,  36,  39,  40,  42,  44,  45,  48,  49,  50,  52,  54,  55,
      56,  60,  63,  64,  65,  66,  70,  72,  75,  77,  78,  80,  84,  88,
      90,  91,  96,  98,  99,  100, 104, 105, 108, 110, 112, 117, 120, 125,
      126, 130, 132, 135, 140, 144, 147, 150, 154, 156, 160, 165, 168, 175,
      176, 180, 182, 189, 192, 195, 196, 198, 200, 208, 210, 216, 220, 224,
      225, 231, 234, 240, 245, 250, 252, 260, 264, 270, 273, 275, 280, 288,
      294, 297, 300, 308, 312, 315, 320, 325, 330, 336, 343, 350, 351, 352,
      360, 364, 375, 378, 385, 390, 392, 396, 400, 416, 420, 432, 440, 441,
      448, 450, 455, 462, 468, 480, 490, 495, 500, 504, 512};

  // select closest (larger) size for given grid size
  for (int i = 0; i < N_GRID_SIZE; i++)
    if ((nGrid + 2) <= listGridSize[i])
      return listGridSize[i] - 2;

  return listGridSize[N_GRID_SIZE - 1] - 2;
}
