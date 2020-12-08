#include <iostream>
#include <map>
#include <random>
#include <set>
#include <unordered_map>
#include <iostream>
#include <fstream>
#define ANSI_COLOR_RED     "\x1b[31m"
using namespace std;
float get_random() {
  static std::default_random_engine e;
  static std::uniform_real_distribution<> dis(0, 1); // rage 0 - 1
  return dis(e);
}

int get_random_int(int max) {
  static std::default_random_engine e;
  std::uniform_int_distribution<int> dis(0, max-1);
  return dis(e);
}

void generate_candidate_blocks(int R, int C, int BS_R, int BS_C, int num_blocks,
                               int *weight_indptr, int *weight_indices) {
  std::map<int, std::set<int>> blocks;
  int num_r_block = R ;
  int num_c_block = C;
  int curr_size = 0;
  while (curr_size < num_blocks) {
    int r = get_random_int(num_r_block);
    int c = get_random_int(num_c_block);
    if (blocks[r].count(c) == 0) {
      blocks[r].insert(c);
      curr_size++;
    }
  }

  int current_ptr = 0;
  int i;
  for (i = 0; i < num_r_block; i++) {
    weight_indptr[i] = current_ptr;
    for (auto block : blocks[i]) {
      weight_indices[current_ptr++] = block;
    }
  }
  weight_indptr[i] = current_ptr;
}


void serial(float *val, int *block_ptr, int *col_ind, float *x,
                 float *y, int n, int bs){
for (int i = 0; i < n; i++) {
  int block_first = block_ptr[i];
  int block_last = block_ptr[i + 1];
  for (int block = block_first; block < block_last; block++) {
    for (int row = 0; row < bs; row++) {
      for (int col = 0; col < bs; col++) {
        //printf("%d  %d  %f \n", i * bs + row,
               int row_v=i*bs+row;
               int column =col_ind[block] * bs + col;
               y[row_v]+=val[block * bs * bs + row * bs + col]*x[column];
      }
    }
  }
}
}


void test_spmv(float *val, int *block_ptr, int *col_ind, float *x, int n,
               int blockSize, int num_blocks,int rows) {
//float *ysparse=(float* )malloc(sizeof(float)*rows);
//compute_BSR(val,block_ptr,col_ind,x,y,n,block_size,num_blocks);
float *yserial=(float* )malloc(sizeof(float)*rows);
for(int i=0;i<rows;i++){
  yserial[i]=0;
}
serial(val,block_ptr,col_ind,x,yserial,n,blockSize);
for(int i=0;i<rows;i++){
  printf("%.1f ",yserial[i]);
}
printf("\n" );
//float* ybsr1=(float *)malloc(sizeof(float)*rows);
//bsr1run(val,block_ptr,col_ind,x,y,n,block_size,num_blocks,bsr1run);
               }

int main(int argc, char **argv) {

  ofstream myfile;
  myfile.open ("matrix.txt");
  int N = 1 << atoi(argv[1]);
  int K = N;
  int bs = 1 << atoi(argv[2]);
  myfile<<"N= "<<N<<" bs= "<<bs<<"\n";
  float density =(float)1/( 1<<(atoi(argv[3])));
  printf("density=%f\n",density );
  float *data;
  float *weight;
  int *weight_ind;
  int *weight_ptr;
  int nnz = int(density * K * N*bs*bs);
  printf("nnz=%d\n",nnz );
  int num_blocks = int(nnz / (bs * bs)) + 1;
  printf("num_blocks=%d\n",num_blocks );
  weight = (float *)malloc(num_blocks * bs * bs * sizeof(float));
  weight_ind = (int *)malloc(num_blocks * sizeof(int));
  weight_ptr = (int *)malloc((N + 1) * sizeof(int));


  for (int i = 0; i < num_blocks * bs * bs; i++) {
    weight[i] = 10*get_random();
  }

  generate_candidate_blocks(N, K, bs, bs, num_blocks, weight_ptr,
                            weight_ind);
  int rows=N*bs;

  float *x =(float *) malloc(rows * sizeof(float));
  for(int i=0;i<rows;i++){x[i]=1;}
  test_spmv( weight, weight_ptr, weight_ind, x, N,bs,num_blocks,rows);
/*
  float** bsr=new float*[rows];
  for(int i = 0; i < rows; ++i)
      bsr[i] = new float[rows];
  for(int i=0;i<rows;i++){
    for(int j=0;j<rows;j++){
      bsr[i][j]=0;
    }
  }
*/
  int nnzcntr=0;
    for (int i = 0; i < N; i++) {
      int block_first = weight_ptr[i];
      int block_last = weight_ptr[i + 1];
      for (int block = block_first; block < block_last; block++) {
        for (int row = 0; row < bs; row++) {
          for (int col = 0; col < bs; col++) {
            myfile <<  i * bs + row <<" "<<weight_ind[block] * bs + col<<" "<<weight[block * bs * bs + row * bs + col]<<"\n";
            if(weight_ind[block] * bs + col>=rows){
              cout <<"Error "<<  i * bs + row <<" "<<weight_ind[block] * bs + col<<" "<<weight[block * bs * bs + row * bs + col]<<"\n";

            }
            nnzcntr++;
            //bsr[ i * bs + row][weight_ind[block] * bs + col]=weight[block * bs * bs + row * bs + col];
          }
        }
      }
    }
    printf("nnzcntr=%d\n",nnzcntr );
/*
    for(int i=0;i<rows;i++){
      for(int j=0;j<rows;j++){
        if(bsr[i][j]>0){
            printf("x " );
          }
            else{
          printf("o " );}


      }
      printf("\n" );
    }*/

  myfile.close();
  return 0;
}
