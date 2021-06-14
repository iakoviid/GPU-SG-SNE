cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sg_tsneCUDA)
set (CMAKE_PROJECT_VERSION 2)
set (CMAKE_PROJECT_VERSION_MAJOR 2)
set (CMAKE_PROJECT_VERSION_MINOR 1)
set (CMAKE_PROJECT_VERSION_PATH 0)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_REQUIRED_FLAGS "-fcilkplus -lmetis -ltbbmalloc  -lfftw3_threads -lfftw3")
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC -m64 -std=c++11 -mtune=native -fopenmp -fcilkplus  -DALIGN=64")
SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -lflann -lmetis -ltbbmalloc -lfftw3_threads -lfftw3  -lcilkrts")
# CUDA Configuration
#-------------------------------------------------------------------------------
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
SET(CUDA_HOST_COMPILER g++)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    -O2
                    -Xptxas -dlcm=cg
                    -gencode=arch=compute_60,code=sm_60
                    -gencode=arch=compute_61,code=sm_61
                    -gencode=arch=compute_70,code=sm_70
                    -std=c++14
                    -Xcompiler '-O2'
                    -Xcompiler '-fPIC'
                    -Xcompiler '-fopenmp'
                    -Xcompiler '-msse4'
                    -Xcompiler '-m64'
                    -Xcompiler '-mpopcnt'
#                  -Xcompiler '-g'
       # -fmad=false
                #       -m64
#               -G
                    --default-stream per-thread
                #  --use_fast_math
                    -maxrregcount=24

                    )
get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})

include_directories(
  ./csb
  ${CUDA_INCLUDE_DIRECTORIES}
  ./src
  ./src/cuda/

)


set(SOURCES_CPU
	./src/sgtsne.cpp
	./src/utils.cpp
	./src/sparsematrix.cpp
	./csb/csb_wrapper.cpp
	./src/graph_rescaling.cpp
	./src/nuconv.cpp
	./src/timers.cpp
	./src/pq.cpp
	./src/gradient_descendHYB.cpp
	./src/qq.cpp
	./src/demo_perplexity_equalization.cpp
#	./csb/bandMat.cpp
#	./csb/bicsb.cpp
#	./csb/matfileio.cpp
#	./csb/csc.cpp
#	./csb/bmcsb.cpp
#	./csb/bmsym.cpp
#	./csb/csr_routines.cpp
#	./csb/utils.cpp
#	./csb/spvec.cpp
#	./csb/bssb.cpp
#	./csb/csbsym.cpp
#	./csb/SSEspmv.cpp
#	./csb/matfileio_all.cpp

)

set(SOURCES_GPU
        ./src/cuda/nuconv.cu
        ./src/cuda/Frep.cu
        ./src/cuda/non_periodic_convD.cu
        ./src/cuda/non_periodic_convF.cu
        ./src/cuda/gridding.cu
)


cuda_add_library(gpu_obj ${SOURCES_GPU})
cuda_add_executable(tsne ${SOURCES_CPU})
target_link_libraries(tsne  gpu_obj -O2 -fPIC -m64 -std=c++11 -mtune=native
  -fopenmp -fcilkplus -DALIGN=64 ${CUDA_CUFFT_LIBRARIES}  ${CUDA_LIBRARIES} -lflann -lmetis -ltbbmalloc -lfftw3_threads -lfftw3  -lcilkrts)
#target_link_libraries(tsne  gpu_obj )
