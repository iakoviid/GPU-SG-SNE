#include "mex.h"
#include "add_wrapper.hpp"

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    // input validation
    if (nrhs != 2 || nlhs > 1)
        mexErrMsgTxt("Wrong number of input/output arguments.");
    if (!mxIsSingle(prhs[0]) || !mxIsSingle(prhs[1]))
        mexErrMsgTxt("Inputs must be single arrays.");
    if (mxIsComplex(prhs[0]) || mxIsComplex(prhs[1]))
        mexErrMsgTxt("Inputs must be real arrays.");
    if (mxIsSparse(prhs[0]) || mxIsSparse(prhs[1]))
        mexErrMsgTxt("Inputs must be dense arrays.");
    if (mxGetNumberOfElements(prhs[0]) != mxGetNumberOfElements(prhs[1]))
        mexErrMsgTxt("Inputs must have the same size.");

    // create ouput array
    mwSize numel = mxGetNumberOfElements(prhs[0]);
    mwSize ndims = mxGetNumberOfDimensions(prhs[0]);
    const mwSize *dims = mxGetDimensions(prhs[0]);
    plhs[0] = mxCreateNumericArray(ndims, dims, mxSINGLE_CLASS, mxREAL);

    // get pointers to data
    float *c = (float*) mxGetData(plhs[0]);
    float *a = (float*) mxGetData(prhs[0]);
    float *b = (float*) mxGetData(prhs[1]);

    // perform addition on the GPU: c = a + b
    addWithCUDA(c, a, b, numel);
}
