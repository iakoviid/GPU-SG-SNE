#include <stdio.h>
#include <stdlib.h>
#define WSIZE 32
#define LOOPS 100000
#define UPPER_BIT 31
#define LOWER_BIT 0

__device__ unsigned int ddata[WSIZE];

// naive warp-level bitwise radix sort

__global__ void mykernel(){
  __shared__ volatile unsigned int sdata[WSIZE*2];
  // load from global into shared variable
  sdata[threadIdx.x] = ddata[threadIdx.x];
  unsigned int bitmask = 1<<LOWER_BIT;
  unsigned int offset  = 0;
  unsigned int thrmask = 0xFFFFFFFFU << threadIdx.x;
  unsigned int mypos;
  //  for each LSB to MSB
  for (int i = LOWER_BIT; i <= UPPER_BIT; i++){
    unsigned int mydata = sdata[((WSIZE-1)-threadIdx.x)+offset];
    unsigned int mybit  = mydata&bitmask;
    // get population of ones and zeroes (cc 2.0 ballot)
    unsigned int ones = __ballot(mybit); // cc 2.0
    unsigned int zeroes = ~ones;
    offset ^= WSIZE; // switch ping-pong buffers
    // do zeroes, then ones
    if (!mybit) // threads with a zero bit
      // get my position in ping-pong buffer
      mypos = __popc(zeroes&thrmask);
    else        // threads with a one bit
      // get my position in ping-pong buffer
      mypos = __popc(zeroes)+__popc(ones&thrmask);
    // move to buffer  (or use shfl for cc 3.0)
    sdata[mypos-1+offset] = mydata;
    // repeat for next bit
    bitmask <<= 1;
    }
  // save results to global
  ddata[threadIdx.x] = sdata[threadIdx.x+offset];
  }

int main(){

  unsigned int hdata[WSIZE];
  for (int lcount = 0; lcount < LOOPS; lcount++){
    unsigned int range = 1U<<UPPER_BIT;
    for (int i = 0; i < WSIZE; i++) hdata[i] = rand()%range;
    cudaMemcpyToSymbol(ddata, hdata, WSIZE*sizeof(unsigned int));
    mykernel<<<1, WSIZE>>>();
    cudaMemcpyFromSymbol(hdata, ddata, WSIZE*sizeof(unsigned int));
    for (int i = 0; i < WSIZE-1; i++) if (hdata[i] > hdata[i+1]) {printf("sort error at loop %d, hdata[%d] = %d, hdata[%d] = %d\n", lcount,i, hdata[i],i+1, hdata[i+1]); return 1;}
    // printf("sorted data:\n");
    //for (int i = 0; i < WSIZE; i++) printf("%u\n", hdata[i]);
    }
  printf("Success!\n");
  return 0;
}
