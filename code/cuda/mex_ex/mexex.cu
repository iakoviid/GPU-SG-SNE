#include "mex.h"
#include "cuda_runtime.h"


__global__ void VectorScalarAddition(const double* A,const double b, double* C, const int N){
	int i=blockDim.x*blockIdx.c+threadIdx.x;
	if(i<N){
		C[i]=A[i]+b;	
	}

}
void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]){

	const double *A;
	double *deviceA;
	double b;
	double *C;
	double *deviceC;
	size_t N;
	char const * const errId= "parallel:gpu:MexVectorScalarAdditionGPU:InvalidInput";
	char const * const erMeg="inv";
	
	if(nrls !=2|| !mxIsDouble(prhs[0] ||mxIsComplex(prhs[0])){
	
	}
	A=mxGetPr(prhs[0]);
	b=mxGetScalar(prhs[1]);
	N=mxGetN(prhs[0]);
	plhs[0]=mxCreateDoubleMatrix(1,(mwSize)N,mxREAL);
	C=mxGetPr(plh[0]);
	cudaMalloc(&deviceA,sizeof(double)*(int)N);
	cudaMalloc(&deviceC,sizeof(double)*(int)N);	
	cudaMemcpy(deviceA,A,(int)N*sizeof(double),cudaMemcpyHostToDevice);

	int threads=256;
	int blocks=(N+threads-1)/threads;
	
	VectorScalarAddition<<<blocks,threads>>>(deviceA,b,deviceC,(int)N);
	cudaMemcpy(C,deviceC,(int)N*sizeof(double),cudaMemcpyDeviceToHost);
	cudaFree(deviceA);
	cudaFree(deviceC);


}
