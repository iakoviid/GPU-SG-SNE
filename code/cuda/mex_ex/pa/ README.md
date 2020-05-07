Example of using CUDA in a MATLAB MEX-function.

Tested on Windows 8.1 x64, MATLAB R2015a, CUDA 6.5, Visual Studio 2013.

Steps to compile and test:

``` bat
C:\> call "c:\Program Files (x86)\Microsoft Visual Studio 12.0\VC\vcvarsall.bat" amd64
C:\> set CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5
C:\> set PATH=%PATH%;%CUDA_PATH%\bin;%CUDA_PATH%\lib;%CUDA_PATH%\lib64
C:\> nvcc -c -m64 add.cu
C:\> matlab
```

``` matlab
>> mex -largeArrayDims main_mex.cpp add.obj -L"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v6.5\lib\x64" -lcudart
>> a = rand([5 5], 'single'); b = rand([5 5], 'single');
>> c1 = a+b;
>> c2 = main_mex(a,b)
```
