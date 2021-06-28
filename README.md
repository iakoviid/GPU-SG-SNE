# GPU-SG-SNE
## A GPU and a Hybrid GPU-CPU implementation of the t-SNE Algorithm. 

## Abstract
t-distributed Stochastic Neighborhood Embedding (t-SNE) is a widely used dimensionality reduction technique, that is particularly well suited for visualization of high-dimensional datasets. On this diploma thesis we introduce a high performance GPU-accelerated implementation of the t-SNE method in CUDA. We base our approach on the work “Spaceland Embedding of Sparse Stochastic Graphs, HPEC 2019” [1]. Obtaining an embedding essentially requires two steps, namely a dense and a sparse computation. One of the main bottlenecks is that usually sparse computation does not scale well in GPU’s because of the irregularity of the memory accesses. To overcome this problem, we use a more suitable sparse matrix storage format that leads to locally dense data and is better suited for GPU processing. The dense computation is performed with an interpolation-based fast Fourier Transform accelerated method. Finally, for high performance multicore systems we introduce a Hybrid CPU-GPU implementation that executes the sparse computation on the CPU and the dense computation on the GPU in parallel, hiding some of the total temporal cost in the process. 

We name the GPU-CUDA implementation of the algorthm SG-tSNE-CUDA and the Hybrid CPU-GPU implementation of the algorithm SG-tSNE-HYB.

### Prerequisites 

SG-tSNE-CUDA uses the following open-source software:

-   [FLANN](https://www.cs.ubc.ca/research/flann/) 1.9.1
-   [CUDA](https://developer.nvidia.com/cuda-downloads) 9.2
-   [CMAKE](https://cmake.org/download/) 3.0

And SG-tSNE-HYB uses the following open-source software:

-   [FFTW3](http://www.fftw.org/) 3.3.8
-   [METIS](http://glaros.dtc.umn.edu/gkhome/metis/metis/overview) 5.1.0
-   [FLANN](https://www.cs.ubc.ca/research/flann/) 1.9.1
-   [Intel TBB](https://01.org/tbb) 2019
-   [CUDA](https://developer.nvidia.com/cuda-downloads) 9.2
-   [CMAKE](https://cmake.org/download/) 3.0

## Usage


## Performance
We precent experiments for random sampled subsets of the 1.3 million element dataset of mice brain cell data (avaliable [here](https://support.10xgenomics.com/single-cell-gene-expression/datasets)). With this proccess we can see how the performance is affected by the number of elements.
![](https://github.com/iakoviid/GPU-SG-SNE/blob/master/images/im1.jpg)
We can see that our implementations are faster than t-SNE-CUDA [2]. With speed up: 
![](https://github.com/iakoviid/GPU-SG-SNE/blob/master/images/im2.jpg)

For the smaller dataset of Mnist (60 thousand elements) we see similar results.
![](https://github.com/iakoviid/GPU-SG-SNE/blob/master/images/im3.jpg)
![](https://github.com/iakoviid/GPU-SG-SNE/blob/master/images/im4.jpg)

As we see the Hybrid implementation is most efficient when we need to embedd large datasets with a multicore machine. 
More detailed analysis of the preformance and implementation methods are included in the thesis and presentation.

## References

[1] Dimitris Floros, Alexandros Stavros Iliopoulos, Nikos Pitsianis and Xiaobai Sun, "Spaceland Embedding of Sparse Stochastic Graph", 2019 IEEE High Performance Extreme Computing Conference (HPEC)

[2] David M. Chan, Roshan Rao, Forrest Huang and John F. Canny, "t-SNE-CUDA: GPU-Accelerated t-SNE and its Applications to Modern Data", 2018.

[3] van der Maaten, Laurens and Hinton, Geoffrey, "Visualizing Data using t-SNE", Journal of Machine Learning Research, vol. 9, pp. 2579–2605, 200

[4] G. C. Linderman, M. Rachh, J. G. Hoskins, S. Steinerberger, and Y. Kluger, “Efficient  Algorithms  for  t-distributed  Stochastic  Neighborhood  Embed-ding,” CoRR, vol. abs/1712.09005, 2017



