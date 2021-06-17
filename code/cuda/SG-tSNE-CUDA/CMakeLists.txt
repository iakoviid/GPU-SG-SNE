cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(sg_tsneCUDA)
set (CMAKE_PROJECT_VERSION 2)
set (CMAKE_PROJECT_VERSION_MAJOR 2)
set (CMAKE_PROJECT_VERSION_MINOR 1)
set (CMAKE_PROJECT_VERSION_PATH 0)
set (CMAKE_SKIP_RULE_DEPENDENCY TRUE)
set (CMAKE_EXPORT_COMPILE_COMMANDS 1)
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/Modules)

option(WITH_FAISS_GPU_STANDALONE "Build standalone gpufaiss library ON" OFF)
option(WITH_MKL "Build with MKL if ON (OpenBLAS if OFF)" OFF)
if(NOT WITH_FAISS_GPU_STANDALONE)
      message("-- Not building standalone gpufaiss lib. To build gpufaiss standalone use -DWITH_FAISS_GPU_STANDALONE=ON")
endif()
SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

# CUDA Configuration
#-------------------------------------------------------------------------------
SET(CUDA_SEPARABLE_COMPILATION ON)
find_package(CUDA REQUIRED)
SET(CUDA_HOST_COMPILER  g++-7)
SET(CUDA_PROPAGATE_HOST_FLAGS OFF)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
                    -O3
                    -Xptxas -dlcm=cg
                    -gencode=arch=compute_60,code=sm_60
                    -gencode=arch=compute_61,code=sm_61
                    -gencode=arch=compute_70,code=sm_70
                    -std=c++14
                    -Xcompiler '-O3'
                    -Xcompiler '-fPIC'
                    -Xcompiler '-fopenmp'
                    -Xcompiler '-msse4'
                    -Xcompiler '-m64'
                    -Xcompiler '-mpopcnt'
                  -Xcompiler '-g'
       # -fmad=false
		#	-m64
              # -G
                  #  --default-stream per-thread
                #  --use_fast_math
		  #  -maxrregcount=24

                    )
get_filename_component(CUDA_LIB_PATH ${CUDA_CUDART_LIBRARY} DIRECTORY)
find_library(CUDA_cudadevrt_LIBRARY cudadevrt PATHS ${CUDA_LIB_PATH})


# BLAS configuration
#-------------------------------------------------------------------------------
if(WITH_MKL)
    find_package(MKL REQUIRED)
    include_directories(${MKL_INCLUDE_DIRS})
    set(BLAS_LIB ${MKL_LIBRARIES})
else()
    find_package(OpenBLAS REQUIRED)
    include_directories(${OpenBLAS_INCLUDE_DIR})
    set(BLAS_LIB ${OpenBLAS_LIB})
endif()

add_subdirectory(third_party/faiss)


#non constant grid init faiss
#-------------------------------------------------------------------------------
# FAISS Configuration
#-------------------------------------------------------------------------------
if(WITH_FAISS_GPU_STANDALONE)
    add_subdirectory(third_party/faiss/gpu)
endif()

# specify header and cpp files
file(GLOB_RECURSE faiss_gpu_headers third_party/faiss/gpu/*.h)
file(GLOB_RECURSE faiss_gpu_cpp third_party/faiss/gpu/*.cpp)
file(GLOB_RECURSE faiss_gpu_cuh third_party/faiss/gpu/*.cuh)
file(GLOB_RECURSE faiss_gpu_cu third_party/faiss/gpu/*.cu)

# Project Setup
#-------------------------------------------------------------------------------
include_directories(
  src/
  ${CUDA_INCLUDE_DIRECTORIES}
  third_party/faiss/
  third_party/
  ../cusplibrary
)
link_directories(
  ${CUDA_LIB_PATH}
)

set(SG_tSNE_CUDA_SOURCES

    src/testing_modules/demo_perplexity.cu
    src/perplexity_search.cu

    src/sgtsne.cu
    src/gradient_descend.cu
    src/utils.cu
    src/pq.cu
    src/matrix_converter.cpp
    src/Frep.cu
    src/graph_rescaling.cu
    src/gridding.cu
    src/non_periodic_convF.cu
    src/non_periodic_convD.cu
    src/nuconv.cu
)
cuda_add_executable(sg_tsne_cuda
    ${faiss_gpu_headers} ${faiss_gpu_cpp} ${faiss_gpu_cuh} ${faiss_gpu_cu}
    ${SG_tSNE_CUDA_SOURCES}
)
target_link_libraries(sg_tsne_cuda
            ${CUDA_LIBRARIES}
            ${CUDA_CUFFT_LIBRARIES}
            ${CUDA_cusparse_LIBRARY}
                    ${CUDA_CUBLAS_LIBRARIES}
                    faiss
                    ${BLAS_LIB}

	           -lcusolver -lflann
             -L/usr/local/cuda/lib64/ -lcudadevrt -lcudart

)
#------------------------------------------------------
set(SG_tSNE_CUDA_VARGRID_SOURCES

    src/testing_modules/demo_perplexity.cu
    src/perplexity_search.cu

    src/sgtsne.cu
    src/gradient_descend_varGrid.cu
    src/utils.cu
    src/pq.cu
    src/matrix_converter.cpp
    src/Frep_varGrid.cu
    src/graph_rescaling.cu
    src/gridding.cu
    src/non_periodic_convF.cu
    src/non_periodic_convD.cu
    src/nuconv.cu
)
cuda_add_executable(demo_perplexity_nonconst
    ${faiss_gpu_headers} ${faiss_gpu_cpp} ${faiss_gpu_cuh} ${faiss_gpu_cu}
    ${SG_tSNE_CUDA_VARGRID_SOURCES}
)
target_link_libraries(demo_perplexity_nonconst
            ${CUDA_LIBRARIES}
            ${CUDA_CUFFT_LIBRARIES}
            ${CUDA_cusparse_LIBRARY}
                    ${CUDA_CUBLAS_LIBRARIES}
                    faiss
                    ${BLAS_LIB}

                   -lcusolver -lflann
             -L/usr/local/cuda/lib64/ -lcudadevrt -lcudart

)
