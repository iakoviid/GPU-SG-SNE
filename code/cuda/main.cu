#include <stdio.h>
#include <stdlib.h>
#include "mmio.h"
#include "mmio.c"
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/random.h>
#include <thrust/sort.h>
#include <time.h>


#define CUDA_CALL(x)                                                           \
  {                                                                            \
    if ((x) != cudaSuccess) {                                                  \
      printf("CUDA error at %s:%d\n", __FILE__, __LINE__);                     \
      printf("  %s\n", cudaGetErrorString(cudaGetLastError()));                \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  }

/*Size of shared memory that is used has to be greater than the max( sum of lengths of the columns)*sizeof(int) that we store*/
#define sharedsize 512


/*Making the col_ptr array st if i'th column is nonempty then col_ptr[i]=Start of i'th column 
and col_ptr[i+1]-col_ptr[i] = length of i'th column*/
__global__ void findCol_ptr(int *dJ, int nz, int *col_ptr) {
  
 for(int i = blockIdx.x * blockDim.x + threadIdx.x+1; i<nz;i+=gridDim.x*blockDim.x){
     if(i<nz){
    int x=dJ[i];
    int y=dJ[i-1];
    if (x != y) {
      col_ptr[x] = i;
    
    if(y+1!=x){
        col_ptr[y + 1] = i;
    }}
    if(i==nz-1){
        col_ptr[x+1]=nz;
    }
    if(i==1){
    col_ptr[0]=0;

  }
  }
  } 
}

/*Unrolled sum reduction of a Warp*/
__device__ void warpReduce(volatile int* sdata,int tid ){
    sdata[tid]+=sdata[tid +32];
    sdata[tid]+=sdata[tid +16];
    sdata[tid]+=sdata[tid +8];
    sdata[tid]+=sdata[tid +4];
    sdata[tid]+=sdata[tid +2];
    sdata[tid]+=sdata[tid +1];
}
/*Counts the number of triangles for the j element in the blockCol array*/
__device__ int ComputeIntersection(int* blockCol,int* col_ptr,int len,int nz,int* dI,int j){    
        int k1;
        int k2;
        int s=0;
        k1=j+1;
        int x=blockCol[j];
        k2=col_ptr[x];
        int r1;
        int r2;
        if(k2>0){
        int len2=col_ptr[x+1];
        r1=blockCol[k1];
        r2=dI[k2];
        while(k1<len && k2<len2 ) {
            if(r1==r2){
                s++;      
                k1++;
                k2++;
                if(k2==nz || k1==len ){break;}

                r1=blockCol[k1];
                r2=dI[k2];

       

            }else if(r1>r2){
                k2++;
                if(k2==nz){break;}

                r2=dI[k2];

            }else{
            k1++;
            if(k1==len ){break;}

            r1=blockCol[k1];

            }
            
        }}


        
        return s;
    }
/*Block counts the number of triangles for a number of columns*/

__global__ void computeCol(int* dI,int* dJ,int nz,int* col,int* out, int N,int k) {
    int s=0;
    int j;
    extern __shared__ int nt[];  
     int* blockCol=&nt[blockDim.x];
    int tid=threadIdx.x;
    for(int i=blockIdx.x;i<N/k+1;i+=gridDim.x){
    /*Find the lengths of k consequtive columns */
        
        int a=0;
        int b=-1;
        int len;
        int colStart;
        for(j=0;j<k;j++){

	        
	        if(k*i+j<N-1){

		        colStart=col[k*i+j];
		        len= col[k*i+1+j]-colStart;
	        }
	        else{len= 0;}

	        if(colStart>=0 && len>0){
	            if(tid<32){
	            nt[j]=len;}
	            a=a+len;
	            if(b==-1){

	                b=colStart;
	            }
	        }else{ 
	            len=0;
	             if(tid<32){
	            nt[j]=len;}


	        }
        }
                  __syncthreads();

      /*Load the columns*/
        colStart=b;
        for( j=tid;j<a;j+=blockDim.x)
        {       


                blockCol[j]=dI[j+colStart];
                
        }
          __syncthreads();

     /*Search each element of the columns*/
     for( j=tid;j<a;j+=blockDim.x)
        {   
        	/*Find the apropriate length in the blockCol array*/
            b=0;
            for(int x=0;x<k;x++){
                b=b+nt[x];
                if(j<b){break;}
            }
        s=s+ComputeIntersection(blockCol,col,b ,nz, dI, j);
     }


      
      
    }
    /*Sum redusce the results of the threads*/
              nt[tid]=s;
        __syncthreads();


        //do reduction in shared mem 
        for( s=blockDim.x/2; s>32;s>>=1){
            if(tid<s){
            nt[tid]+=nt[tid+s];
            }
        __syncthreads();
        }
    if(tid<32){ warpReduce(nt,tid);}

        if(tid<32){
        out[blockIdx.x]+=nt[0];}


}

void ReadMatrix(int* M,int* N,int* nz,int** I,int** J,int argc,char** argv){

    int i;
    int ret_code;
        MM_typecode matcode;
        FILE *f;
        
        if (argc < 4)
        {
            fprintf(stderr, "Usage: %s [martix-market-filename] [threadsPerBlock] [numberOfBlocks]\n", argv[0]);
            exit(1);
        }
        else    
        { 
            if ((f = fopen(argv[1], "r")) == NULL) 
                exit(1);
        }

        if (mm_read_banner(f, &matcode) != 0)
        {
            printf("Could not process Matrix Market banner.\n");
            exit(1);
        }

        if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
                mm_is_sparse(matcode) )
        {
            printf("Sorry, this application does not support ");
            printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
            exit(1);
        }

        /* find out size of sparse matrix .... */

        if ((ret_code = mm_read_mtx_crd_size(f, M, N, nz)) !=0)
            exit(1);

    /* reseve memory for matrices */

    *I = (int *) malloc(*nz * sizeof(int));
    *J = (int *) malloc(*nz * sizeof(int));
  

    for (i=0; i<*nz; i++)
    {
        fscanf(f, "%d %d\n", &(*I)[i], &(*J)[i]);
        (*I)[i]--;  /* adjust from 1-based to 0-based */
        (*J)[i]--;
    }

    if (f !=stdin) fclose(f);
}

int klength(int* col,int N,int k){
    int maxlen=0;
    int len;
    for(int i=0;i<N/k+1;i++){
        int a=0;
        for(int j=0;j<k;j++){
            if(k*i+k<N){
        len=col[k*i+1+j]-col[k*i+j];
    
        if(col[i*k+j]>=0 && len>0){
        a=a+len;
            if(a>maxlen){
                maxlen=a;
            }
        }}
    }
    }
    return maxlen;
}
int main(int argc, char *argv[])
{

    int M, N, nz;   
    int *I, *J;

    ReadMatrix(&M,&N,&nz,&I,&J,argc,argv);

   //mm_write_banner(stdout, matcode);
    //printf("nz=%d M=%d N=%d\n",nz,M,N);

    /*Arguments k number of consequtive columns processed by each block and kernel launching parameters threadsPerBlock Blocks*/
    int k=atoi(argv[4]);
    int threadsPerBlock=atoi(argv[2]);
    int Blocks=atoi(argv[3]);
    int length=atoi(argv[5]);
    /*Device data*/
    int* dI;
    int* dJ;
    int* col;
    int* out;
    CUDA_CALL(cudaMalloc(&dI, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&dJ, nz*sizeof(int)));
    CUDA_CALL(cudaMalloc(&col, N*sizeof(int)));
    CUDA_CALL(cudaMalloc(&out, Blocks*sizeof(int)));


    cudaMemcpy(dI, I, nz*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dJ, J, nz*sizeof(int), cudaMemcpyHostToDevice);

    //int length=klength(col,N,k);
    float time;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    CUDA_CALL(cudaMemset(col, -1, N* (sizeof(int))));
    CUDA_CALL(cudaMemset(out, 0, Blocks* (sizeof(int))));


    findCol_ptr<<<Blocks,threadsPerBlock>>>(dJ,nz,col);
    
    computeCol<<<Blocks,threadsPerBlock,(threadsPerBlock+length)*sizeof(int)>>>(dI,dJ,nz,col,out,N,k);
      
    
    thrust::device_ptr<int> outptr(out);
    int tot =thrust::reduce(outptr, outptr + Blocks); 
    

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("%f  ms ", time);





    printf(" Trianles =  %d\n",tot );
   

    CUDA_CALL(cudaFree(out));
    CUDA_CALL(cudaFree(dI));
    CUDA_CALL(cudaFree(dJ));
    CUDA_CALL(cudaFree(col));






    return 0;
}
