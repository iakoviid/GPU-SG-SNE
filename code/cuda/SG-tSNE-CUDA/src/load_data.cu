/*simulation data two clusters */
#include <random>



std::default_random_engine generator;
std::normal_distribution<float> distribution1(-10.0, 1.0);
std::normal_distribution<float> distribution2(10.0, 1.0);

thrust::host_vector<float> h_X(dim * num_points);
for (int i = 0; i < dim * num_points; i ++) {
    if (i < ((num_points / 2) * dim)) {
        h_X[i] = distribution1(generator);
    } else {
        h_X[i] = distribution2(generator);
    }
}
