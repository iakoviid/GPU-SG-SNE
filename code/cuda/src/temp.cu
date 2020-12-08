#include <stdio.h>

__global__
static void shflTest(int lid){
    int tid = threadIdx.x;
    float value = tid + 0.1f;
    int* ivalue = reinterpret_cast<int*>(&value);
        if(threadIdx.x==0){value=100.1;}
    //use the integer shfl
    int temp= __shfl_sync(0xFFFFFFFF, ivalue[0],0,32);
    value =reinterpret_cast<float *> (&temp)[0];

    //float x = reinterpret_cast<float*>(&ix)[0];
   // float y = reinterpret_cast<float*>(&iy)[0];

    if(tid == 5){
        printf("value=%f\n",value);
 //printf("shfl tmp %d %d\n",ix,iy);
   //     printf("shfl final %f %f\n",x,y);
    }
}

int main()
{
    shflTest<<<1,32>>>(0);
    cudaDeviceSynchronize();
    return 0;
}
