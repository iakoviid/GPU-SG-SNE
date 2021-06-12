
// FAISS includes
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/GpuIndexIVFFlat.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/IndexProxy.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <stdint.h>

void KNearestNeighbors(int64_t* indices, float* distances,
        const float* const points, const int num_dims,
        const int num_points, const int num_near_neighbors) {
    const int32_t kNumCells = static_cast<int32_t>(
            std::sqrt(static_cast<float>(num_points)));
    const int32_t kNumCellsToProbe = 20;

        faiss::gpu::StandardGpuResources faiss_resources;
        faiss::gpu::GpuIndexIVFFlatConfig faiss_config;
        faiss_config.device = 0;

        faiss_config.indicesOptions = faiss::gpu::INDICES_32_BIT;
        faiss_config.flatConfig.useFloat16 = false;
        faiss_config.useFloat16IVFStorage = false;

        faiss::gpu::GpuIndexIVFFlat search_index(&faiss_resources, num_dims, kNumCells, faiss::METRIC_L2, faiss_config);
        search_index.setNumProbes(kNumCellsToProbe);

        faiss::gpu::IndexProxy search_proxy;
        search_proxy.addIndex(&search_index);
        search_proxy.train(num_points, points);
        search_proxy.add(num_points, points);
        search_proxy.search(num_points, points, num_near_neighbors, distances, indices);


}

void KNearestNeighborsCompresed(int64_t* indices, float* distances,
        const float* const points, const int num_dims,
        const int num_points, const int num_near_neighbors) {

    const int32_t kNumCells = static_cast<int32_t>(
            std::sqrt(static_cast<float>(num_points)));
    const int32_t kNumCellsToProbe = 20;
    const int32_t kSubQuant = 2;
    const int32_t kBPC = 8;
    //  Construct the GPU resources necessary
     faiss::gpu::StandardGpuResources faiss_resources;
     faiss_resources.noTempMemory();

    // Construct the GPU configuration object
     faiss::gpu::GpuIndexIVFPQConfig faiss_config;
        faiss_config.device = 0;

        faiss_config.indicesOptions = faiss::gpu::INDICES_32_BIT;
        faiss_config.flatConfig.useFloat16 = false;
        faiss_config.usePrecomputedTables = true;
        //faiss_config.useFloat16IVFStorage = false;

        faiss::gpu::GpuIndexIVFPQ search_index(&faiss_resources, num_dims, kNumCells, kSubQuant, kBPC, faiss::METRIC_L2, faiss_config);

        search_index.setNumProbes(kNumCellsToProbe);
        //  Add the points to the index
         search_index.train(num_points, points);
         search_index.add(num_points, points);

         // Perform the KNN query
         search_index.search(num_points, points, num_near_neighbors,distances, indices);


}
#include <flann/flann.h>
void allKNNsearchflann(int * IDX,        //!< [k-by-N] array with the neighbor IDs
                  float * DIST,    //!< [k-by-N] array with the neighbor distances
                  float * dataset, //!< [L-by-N] array with coordinates of data points
                  int N,            //!< [scalar] Number of data points N
                  int dims,         //!< [scalar] Number of dimensions L
                  int kappa) {      //!< [scalar] Number of neighbors k


  struct FLANNParameters p;

  p = DEFAULT_FLANN_PARAMETERS;
  p.algorithm = FLANN_INDEX_KDTREE;
  p.trees = 8;
  p.target_precision=0.9;
  p.checks = 300;

   flann_find_nearest_neighbors_float(dataset, N, dims, dataset, N, IDX, DIST, kappa, &p);

}
void allKNNsearch(int * IDX,        //!< [k-by-N] array with the neighbor IDs
                  float * DIST,    //!< [k-by-N] array with the neighbor distances
                  float * dataset, //!< [L-by-N] array with coordinates of data points
                  int N,            //!< [scalar] Number of data points N
                  int dims,         //!< [scalar] Number of dimensions L
                  int kappa) {      //!< [scalar] Number of neighbors k



  long *knn_indices = (long *)malloc(N * kappa*sizeof(long));
  KNearestNeighbors( knn_indices, DIST, dataset, dims, N, kappa);

  for(int i=0;i<N*kappa;i++)IDX[i]=(int)knn_indices[i];
  free(knn_indices);

}
