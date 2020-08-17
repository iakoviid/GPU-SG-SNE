/*Assuming CSR format*/
#include "pq.cuh"

__global__ void PQKernel(coord *Fattr, coord *const Y, double const *const p_sp,
                         matidx *ir, matidx *jc, int const n, int const d) {

const unsigned int thread_id=blockIdx.x*blockDim.x+threadIdx.x;
const unsigned int warp_id=thread_id/32;
const unsigned int lane=thread_id%32;

 unsigned int row=warp_id;

coord sum1=0;
coord sum2=0;
coord sum3=0;
coord dist=0;
unsigned int column;
coord p_times_q=0;
for (row=warp_id;row<n;row=row +gridDim.x*blockDim.x/32){
  sum1=0;
  sum2=0;
  sum3=0;
  dist=0;
  const unsigned int row_start=ir[row];
  const unsigned int row_end=ir[row+1];
  for(unsigned int element=row_start+lane;element<row_end;element+=32){

    column=jc[element];
    dist=0;

    for(int dim=0;dim<d;dim++){
      dist+=(Y[row+dim*n]-Y[column+dim*n])*(Y[row+dim*n]-Y[column+dim*n]);
    }
    p_times_q=p_sp[element]/(1+dist);
    switch (d) {
      case 1:
        sum1+=p_times_q*(Y[row]- Y[column]);
        break;
      case 2:
        sum1+=p_times_q*(Y[row]-Y[column]);
        sum2+=p_times_q*(Y[row+n]-Y[column+n]);
        break;
      case 3:
        sum1+=p_times_q*(Y[row]-Y[column]);
        sum2+=p_times_q*(Y[row+n]-Y[column+n]);
        sum3+=p_times_q*(Y[row+2*n]-Y[column+2*n]);
        break;
    }


}
switch (d) {
  case 1:
    sum1 =warp_reduce(sum1);
    break;
  case 2:
    sum1 =warp_reduce(sum1);
    sum2=warp_reduce(sum2);
    break;
  case 3:
    sum1 =warp_reduce(sum1);
    sum2=warp_reduce(sum2);
    sum3=warp_reduce(sum3);
    break;

}
if(lane==0 ){
  switch (d) {
    case 1:
      Fattr[row]=sum1;
    case 2:
      Fattr[row]=sum1;
      Fattr[row+n]=sum2;
    case 3:
      Fattr[row]=sum1;
      Fattr[row+n]=sum2;
      Fattr[row+2*n]=sum3;

  }

}


}

}
