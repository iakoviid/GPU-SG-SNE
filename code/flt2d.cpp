
#include "stdlib.h"
#include "stdio.h"
#include <math.h>
#include <complex>
#include <fftw3.h>
#include <cmath>
#include<time.h> 

using namespace std;
double kernel(double x1, double x2, double y1, double y2,double df) {
    return pow(1.0 + pow(x1 - y1, 2) + pow(x2 - y2, 2), -1);
}
void precompute_2d(double x_max, double x_min, double y_max, double y_min, int n_boxes, int n_interpolation_points,
                    double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacings,
                   double *y_tilde, double *x_tilde, complex<double> *fft_kernel_tilde, double df );


void n_body_fft_2d(int N, int n_terms, double *xs, double *ys, double *chargesQij, int n_boxes,
                   int n_interpolation_points, double *box_lower_bounds, double *box_upper_bounds,
                   double *y_tilde_spacings, complex<double> *fft_kernel_tilde, double *potentialQij);
int main(){
int D=2;
int N=1000;
//srand(time(0)); 
    // For convenience, split the x and y coordinate values
double * xs=(double *)malloc(sizeof(double)*N);
double * ys=(double *)malloc(sizeof(double)*N);
double* Y=(double *)malloc(sizeof(double)*2*N);
for(int i=0;i<N;i++){
    scanf("%lf,%lf",&Y[2*i],&Y[2*i+1]);
    xs[i]=Y[2*i];
    ys[i]=Y[2*i+1];
    
    //Y[2*i]((double) rand() / (RAND_MAX));
    //Y[2*i+1]=((double) rand() / (RAND_MAX));


}
double * dC=(double *)malloc(sizeof(double)*N*D);
 // Zero out the gradient
for (int i = 0; i < N * D; i++) dC[i] = 0.0;






    double min_coord = 100000;
    double max_coord = -100000;
   // Find the min/max values of the x and y coordinates
    for (unsigned long i = 0; i < N; i++) {
        xs[i] = Y[i * 2 + 0];
        ys[i] = Y[i * 2 + 1];
        if (xs[i] > max_coord) max_coord = xs[i];
        else if (xs[i] < min_coord) min_coord = xs[i];
        if (ys[i] > max_coord) max_coord = ys[i];
        else if (ys[i] < min_coord) min_coord = ys[i];
    }

 // The number of "charges" or s+2 sums i.e. number of kernel sums
    int n_terms = 3;
    auto *chargesQij = new double[N * n_terms];
    auto *potentialsQij = new double[N * n_terms]();

    // Prepare the terms that we'll use to compute the sum i.e. the repulsive forces
    for (unsigned long j = 0; j < N; j++) {
        chargesQij[j * n_terms + 0] = 1;
        chargesQij[j * n_terms + 1] = xs[j];
        chargesQij[j * n_terms + 2] = ys[j];
    }


double x_max=1;
double y_max=1;
double x_min=0;
double y_min=0;
int n_boxes_per_dim=13;
int n_interpolation_points=5;

  int allowed_n_boxes_per_dim[20] = {25,36, 50, 55, 60, 65, 70, 75, 80, 85, 90, 96, 100, 110, 120, 130, 140,150, 175, 200};
    if ( n_boxes_per_dim < allowed_n_boxes_per_dim[19] ) {
        //Round up to nearest grid point
        int chosen_i;
        for (chosen_i =0; allowed_n_boxes_per_dim[chosen_i]< n_boxes_per_dim; chosen_i++);
        n_boxes_per_dim = allowed_n_boxes_per_dim[chosen_i];
    }

    int n_boxes = n_boxes_per_dim * n_boxes_per_dim;

    auto *box_lower_bounds = new double[2 * n_boxes];
    auto *box_upper_bounds = new double[2 * n_boxes];

    auto *y_tilde_spacings = new double[n_interpolation_points];
    int n_interpolation_points_1d = n_interpolation_points * n_boxes_per_dim;
    auto *x_tilde = new double[n_interpolation_points_1d]();
    auto *y_tilde = new double[n_interpolation_points_1d]();
    auto *fft_kernel_tilde = new complex<double>[2 * n_interpolation_points_1d * 2 * n_interpolation_points_1d];


double df=0;// have no idea what this is

precompute_2d( max_coord, min_coord, max_coord, min_coord, n_boxes,  n_interpolation_points,  box_lower_bounds,  box_upper_bounds,  y_tilde_spacings,
 y_tilde,  x_tilde, fft_kernel_tilde ,df );

 n_body_fft_2d( N,  n_terms,  xs,  ys,  chargesQij,  n_boxes,
                    n_interpolation_points,  box_lower_bounds,  box_upper_bounds,
                    y_tilde_spacings, fft_kernel_tilde, potentialsQij);
  

   // Compute the normalization constant Z or sum of q_{ij}. This expression is different from the one in the original
    // paper, but equivalent. This is done so we need only use a single kernel (K_2 in the paper) instead of two
    // different ones. We subtract N at the end because the following sums over all i, j, whereas Z contains i \neq j
    double sum_Q = 0;
    for (unsigned long i = 0; i < N; i++) {
        double phi1 = potentialsQij[i * n_terms + 0];
        double phi2 = potentialsQij[i * n_terms + 1];
        double phi3 = potentialsQij[i * n_terms + 2];
        double phi4 = potentialsQij[i * n_terms + 3];

        sum_Q += (1 + xs[i] * xs[i] + ys[i] * ys[i]) * phi1 - 2 * (xs[i] * phi2 + ys[i] * phi3) + phi4;
    }
    sum_Q -= N;

        //double *pos_f = new double[N * 2];

    // Make the negative term, or F_rep in the equation 3 of the paper
    double *neg_f = new double[N * 2];
    for (unsigned int i = 0; i < N; i++) {
        neg_f[i * 2 + 0] = (xs[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 1]) / sum_Q;
        neg_f[i * 2 + 1] = (ys[i] * potentialsQij[i * n_terms] - potentialsQij[i * n_terms + 2]) / sum_Q;

        //dC[i * 2 + 0] = pos_f[i * 2] - neg_f[i * 2];
        //dC[i * 2 + 1] = pos_f[i * 2 + 1] - neg_f[i * 2 + 1];
       dC[i * 2 + 0]=- neg_f[i * 2];
          dC[i * 2 + 1]=- neg_f[i * 2+1];
    }

    return 0;



}
void interpolate(int n_interpolation_points, int N, const double *y_in_box, const double *y_tilde_spacings,
                 double *interpolated_values) {
    // The denominators are the same across the interpolants, so we only need to compute them once
    auto *denominator = new double[n_interpolation_points];
    for (int i = 0; i < n_interpolation_points; i++) {
        denominator[i] = 1;
        for (int j = 0; j < n_interpolation_points; j++) {
            if (i != j) {
                denominator[i] *= y_tilde_spacings[i] - y_tilde_spacings[j];
            }
        }
    }
    // Compute the numerators and the interpolant value
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < n_interpolation_points; j++) {
            interpolated_values[j * N + i] = 1;
            for (int k = 0; k < n_interpolation_points; k++) {
                if (j != k) {
                    interpolated_values[j * N + i] *= y_in_box[i] - y_tilde_spacings[k];
                }
            }
            interpolated_values[j * N + i] /= denominator[j];
        }
    }

    delete[] denominator;
}



void precompute_2d(double x_max, double x_min, double y_max, double y_min, int n_boxes, int n_interpolation_points,
                   double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacings,
                   double *y_tilde, double *x_tilde, complex<double> *fft_kernel_tilde, double df ) {
    /*
     * Set up the boxes
     */
    int n_total_boxes = n_boxes * n_boxes;
    double box_width = (x_max - x_min) / (double) n_boxes;

    // Left and right bounds of each box, first the lower bounds in the x direction, then in the y direction
    for (int i = 0; i < n_boxes; i++) {
        for (int j = 0; j < n_boxes; j++) {
            box_lower_bounds[i * n_boxes + j] = j * box_width + x_min;
            box_upper_bounds[i * n_boxes + j] = (j + 1) * box_width + x_min;

            box_lower_bounds[n_total_boxes + i * n_boxes + j] = i * box_width + y_min;
            box_upper_bounds[n_total_boxes + i * n_boxes + j] = (i + 1) * box_width + y_min;
        }
    }

    // Coordinates of each (equispaced) interpolation node for a single box
    double h = 1 / (double) n_interpolation_points;
    y_tilde_spacings[0] = h / 2;
    for (int i = 1; i < n_interpolation_points; i++) {
        y_tilde_spacings[i] = y_tilde_spacings[i - 1] + h;
    }

    // Coordinates of all the equispaced interpolation points
    int n_interpolation_points_1d = n_interpolation_points * n_boxes;
    int n_fft_coeffs = 2 * n_interpolation_points_1d;

    h = h * box_width;
    x_tilde[0] = x_min + h / 2;
    y_tilde[0] = y_min + h / 2;
    for (int i = 1; i < n_interpolation_points_1d; i++) {
        x_tilde[i] = x_tilde[i - 1] + h;
        y_tilde[i] = y_tilde[i - 1] + h;
    }

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    auto *kernel_tilde = new double[n_fft_coeffs * n_fft_coeffs]();
    for (int i = 0; i < n_interpolation_points_1d; i++) {
        for (int j = 0; j < n_interpolation_points_1d; j++) {
            double tmp = kernel(y_tilde[0], x_tilde[0], y_tilde[i], x_tilde[j],df );
            kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d + j)] = tmp;
            kernel_tilde[(n_interpolation_points_1d + i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
            kernel_tilde[(n_interpolation_points_1d - i) * n_fft_coeffs + (n_interpolation_points_1d - j)] = tmp;
        }
    }

    // Precompute the FFT of the kernel generating matrix
    fftw_plan p = fftw_plan_dft_r2c_2d(n_fft_coeffs, n_fft_coeffs, kernel_tilde,
                                       reinterpret_cast<fftw_complex *>(fft_kernel_tilde), FFTW_ESTIMATE);
    fftw_execute(p);

    fftw_destroy_plan(p);
    //delete[] kernel_tilde;
}


void n_body_fft_2d(int N, int n_terms, double *xs, double *ys, double *chargesQij, int n_boxes,
                   int n_interpolation_points, double *box_lower_bounds, double *box_upper_bounds,
                   double *y_tilde_spacings, complex<double> *fft_kernel_tilde, double *potentialQij) {
    int n_total_boxes = n_boxes * n_boxes;
    int total_interpolation_points = n_total_boxes * n_interpolation_points * n_interpolation_points;

    double coord_min = box_lower_bounds[0];
    double box_width = box_upper_bounds[0] - box_lower_bounds[0];

    auto *point_box_idx = new int[N];

    // Determine which box each point belongs to
    for (int i = 0; i < N; i++) {
        auto x_idx = static_cast<int>((xs[i] - coord_min) / box_width);
        auto y_idx = static_cast<int>((ys[i] - coord_min) / box_width);
        // TODO: Figure out how on earth x_idx can be less than zero...
        // It's probably something to do with the fact that we use the single lowest coord for both dims? Probably not
        // this, more likely negative 0 if rounding errors
        if (x_idx >= n_boxes) {
            x_idx = n_boxes - 1;
        } else if (x_idx < 0) {
            x_idx = 0;
        }

        if (y_idx >= n_boxes) {
            y_idx = n_boxes - 1;
        } else if (y_idx < 0) {
            y_idx = 0;
        }
        point_box_idx[i] = y_idx * n_boxes + x_idx;
    }

    // Compute the relative position of each point in its box in the interval [0, 1]
    auto *x_in_box = new double[N];
    auto *y_in_box = new double[N];
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i];
        double x_min = box_lower_bounds[box_idx];
        double y_min = box_lower_bounds[n_total_boxes + box_idx];
        x_in_box[i] = (xs[i] - x_min) / box_width;
        y_in_box[i] = (ys[i] - y_min) / box_width;
    }

  
    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `x` direction
    auto *x_interpolated_values = new double[N * n_interpolation_points];
    interpolate(n_interpolation_points, N, x_in_box, y_tilde_spacings, x_interpolated_values);
    // Compute the interpolated values at each real point with each Lagrange polynomial in the `y` direction
    auto *y_interpolated_values = new double[N * n_interpolation_points];
    interpolate(n_interpolation_points, N, y_in_box, y_tilde_spacings, y_interpolated_values);

    auto *w_coefficients = new double[total_interpolation_points * n_terms]();
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i];
        int box_j = box_idx / n_boxes;
        int box_i = box_idx % n_boxes;
        for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++) {
            for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++) {
                // Compute the index of the point in the interpolation grid of points
                int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                          (box_j * n_interpolation_points) + interp_j;
                for (int d = 0; d < n_terms; d++) {
                    w_coefficients[idx * n_terms + d] +=
                            y_interpolated_values[interp_j * N + i] *
                            x_interpolated_values[interp_i * N + i]; //*
                           // chargesQij[i * n_terms + d];
                }
            }
        }
    }

    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
    auto *y_tilde_values = new double[total_interpolation_points * n_terms]();
    int n_fft_coeffs_half = n_interpolation_points * n_boxes;
    int n_fft_coeffs = 2 * n_interpolation_points * n_boxes;
    auto *mpol_sort = new double[total_interpolation_points];

    // FFT of fft_input
    auto *fft_input = new double[n_fft_coeffs * n_fft_coeffs]();
    auto *fft_w_coefficients = new complex<double>[n_fft_coeffs * (n_fft_coeffs / 2 + 1)];
    auto *fft_output = new double[n_fft_coeffs * n_fft_coeffs]();

    fftw_plan plan_dft, plan_idft;
    plan_dft = fftw_plan_dft_r2c_2d(n_fft_coeffs, n_fft_coeffs, fft_input,
                                    reinterpret_cast<fftw_complex *>(fft_w_coefficients), FFTW_ESTIMATE);
    plan_idft = fftw_plan_dft_c2r_2d(n_fft_coeffs, n_fft_coeffs, reinterpret_cast<fftw_complex *>(fft_w_coefficients),
                                     fft_output, FFTW_ESTIMATE);

    for (int d = 0; d < n_terms; d++) {
        for (int i = 0; i < total_interpolation_points; i++) {
            mpol_sort[i] = w_coefficients[i * n_terms + d];
        }

        for (int i = 0; i < n_fft_coeffs_half; i++) {
            for (int j = 0; j < n_fft_coeffs_half; j++) {
                fft_input[i * n_fft_coeffs + j] = mpol_sort[i * n_fft_coeffs_half + j];
            }
        }

        fftw_execute(plan_dft);

        // Take the Hadamard product of two complex vectors
        for (int i = 0; i < n_fft_coeffs * (n_fft_coeffs / 2 + 1); i++) {
            double x_ = fft_w_coefficients[i].real();
            double y_ = fft_w_coefficients[i].imag();
            double u_ = fft_kernel_tilde[i].real();
            double v_ = fft_kernel_tilde[i].imag();
            fft_w_coefficients[i].real(x_ * u_ - y_ * v_);
            fft_w_coefficients[i].imag(x_ * v_ + y_ * u_);
        }

        // Invert the computed values at the interpolated nodes
        fftw_execute(plan_idft);
        for (int i = 0; i < n_fft_coeffs_half; i++) {
            for (int j = 0; j < n_fft_coeffs_half; j++) {
                int row = n_fft_coeffs_half + i;
                int col = n_fft_coeffs_half + j;

                // FFTW doesn't perform IDFT normalization, so we have to do it ourselves. This is done by dividing
                // the result with the number of points in the input
                mpol_sort[i * n_fft_coeffs_half + j] = fft_output[row * n_fft_coeffs + col] /
                                                       (double) (n_fft_coeffs * n_fft_coeffs);
            }
        }
        for (int i = 0; i < n_fft_coeffs_half * n_fft_coeffs_half; i++) {
            y_tilde_values[i * n_terms + d] = mpol_sort[i];
        }
    }

    fftw_destroy_plan(plan_dft);
    fftw_destroy_plan(plan_idft);
    delete[] fft_w_coefficients;
    delete[] fft_input;
    delete[] fft_output;
    delete[] mpol_sort;

    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
        for(int loop_i=0;loop_i<N;loop_i++){
        int box_idx = point_box_idx[loop_i];
        int box_i = box_idx % n_boxes;
        int box_j = box_idx / n_boxes;
        for (int interp_i = 0; interp_i < n_interpolation_points; interp_i++) {
            for (int interp_j = 0; interp_j < n_interpolation_points; interp_j++) {
                for (int d = 0; d < n_terms; d++) {
                    // Compute the index of the point in the interpolation grid of points
                    int idx = (box_i * n_interpolation_points + interp_i) * (n_boxes * n_interpolation_points) +
                              (box_j * n_interpolation_points) + interp_j;
                    potentialQij[loop_i * n_terms + d] +=
                            x_interpolated_values[interp_i * N + loop_i] *
                            y_interpolated_values[interp_j * N + loop_i] *
                            y_tilde_values[idx * n_terms + d];
                }
            }
        }
    }
    delete[] point_box_idx;
    delete[] x_interpolated_values;
    delete[] y_interpolated_values;
    delete[] w_coefficients;
    delete[] y_tilde_values;
    delete[] x_in_box;
    delete[] y_in_box;
}


void precompute(double y_min, double y_max, int n_boxes, int n_interpolation_points, 
                double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacing, double *y_tilde,
                complex<double> *fft_kernel_vector, double df) {
    /*
     * Set up the boxes
     */
    double box_width = (y_max - y_min) / (double) n_boxes;
    // Compute the left and right bounds of each box
    for (int box_idx = 0; box_idx < n_boxes; box_idx++) {
        box_lower_bounds[box_idx] = box_idx * box_width + y_min;
        box_upper_bounds[box_idx] = (box_idx + 1) * box_width + y_min;
    }

    int total_interpolation_points = n_interpolation_points * n_boxes;
    // Coordinates of each equispaced interpolation point for a single box. This equally spaces them between [0, 1]
    // with equal space between the points and half that space between the boundary point and the closest boundary point
    // e.g. [0.1, 0.3, 0.5, 0.7, 0.9] with spacings [0.1, 0.2, 0.2, 0.2, 0.2, 0.1], respectively. This ensures that the
    // nodes will still be equispaced across box boundaries
    double h = 1 / (double) n_interpolation_points;
    y_tilde_spacing[0] = h / 2;
    for (int i = 1; i < n_interpolation_points; i++) {
        y_tilde_spacing[i] = y_tilde_spacing[i - 1] + h;
    }

    // Coordinates of all the equispaced interpolation points
    h = h * box_width;
    y_tilde[0] = y_min + h / 2;
    for (int i = 1; i < total_interpolation_points; i++) {
        y_tilde[i] = y_tilde[i - 1] + h;
    }

    /*
     * Evaluate the kernel at the interpolation nodes and form the embedded generating kernel vector for a circulant
     * matrix
     */
    auto *kernel_vector = new complex<double>[2 * total_interpolation_points]();
    // Compute the generating vector x between points K(y_i, y_j) where i = 0, j = 0:N-1
    // [0 0 0 0 0 5 4 3 2 1] for linear kernel
    // This evaluates the Cauchy kernel centered on y_tilde[0] to all the other points
    //for (int i = 0; i < total_interpolation_points; i++) {
     //   kernel_vector[total_interpolation_points + i].real(kernel(y_tilde[0], y_tilde[i], df));
    //}
    // This part symmetrizes the vector, this embeds the Toeplitz generating vector into the circulant generating vector
    // but also has the nice property of symmetrizing the Cauchy kernel, which is probably planned
    // [0 1 2 3 4 5 4 3 2 1] for linear kernel
    for (int i = 1; i < total_interpolation_points; i++) {
        kernel_vector[i].real(kernel_vector[2 * total_interpolation_points - i].real());
    }

    // Precompute the FFT of the kernel generating vector
    fftw_plan p = fftw_plan_dft_1d(2 * total_interpolation_points, reinterpret_cast<fftw_complex *>(kernel_vector),
                                   reinterpret_cast<fftw_complex *>(fft_kernel_vector), FFTW_FORWARD, FFTW_ESTIMATE);
    fftw_execute(p);
    fftw_destroy_plan(p);

    delete[] kernel_vector;
}



void nbodyfft(int N, int n_terms, double *Y, double *chargesQij, int n_boxes, int n_interpolation_points,
              double *box_lower_bounds, double *box_upper_bounds, double *y_tilde_spacings, double *y_tilde,
              complex<double> *fft_kernel_vector, double *potentialsQij) {
    int total_interpolation_points = n_interpolation_points * n_boxes;

    double coord_min = box_lower_bounds[0];
    double box_width = box_upper_bounds[0] - box_lower_bounds[0];

    // Determine which box each point belongs to
    auto *point_box_idx = new int[N];
    for (int i = 0; i < N; i++) {
        auto box_idx = static_cast<int>((Y[i] - coord_min) / box_width);
        // The right most point maps directly into `n_boxes`, while it should belong to the last box
        if (box_idx >= n_boxes) {
            box_idx = n_boxes - 1;
        }
        point_box_idx[i] = box_idx;
    }

    // Compute the relative position of each point in its box in the interval [0, 1]
    auto *y_in_box = new double[N];
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i];
        double box_min = box_lower_bounds[box_idx];
        y_in_box[i] = (Y[i] - box_min) / box_width;
    }

    /*
     * Step 1: Interpolate kernel using Lagrange polynomials and compute the w coefficients
     */
    // Compute the interpolated values at each real point with each Lagrange polynomial
    auto *interpolated_values = new double[n_interpolation_points * N];
    interpolate(n_interpolation_points, N, y_in_box, y_tilde_spacings, interpolated_values);

    auto *w_coefficients = new double[total_interpolation_points * n_terms]();
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i] * n_interpolation_points;
        for (int interp_idx = 0; interp_idx < n_interpolation_points; interp_idx++) {
            for (int d = 0; d < n_terms; d++) {
                w_coefficients[(box_idx + interp_idx) * n_terms + d] +=
                        interpolated_values[interp_idx * N + i] * chargesQij[i * n_terms + d];
            }
        }
    }

    // `embedded_w_coefficients` is just a vector of zeros prepended to `w_coefficients`, this (probably) matches the
    // dimensions of the kernel matrix K and since we embedded the generating vector by prepending values, we have to do
    // the same here
    auto *embedded_w_coefficients = new double[2 * total_interpolation_points * n_terms]();
    for (int i = 0; i < total_interpolation_points; i++) {
        for (int d = 0; d < n_terms; d++) {
            embedded_w_coefficients[(total_interpolation_points + i) * n_terms + d] = w_coefficients[i * n_terms + d];
        }
    }

    /*
     * Step 2: Compute the values v_{m, n} at the equispaced nodes, multiply the kernel matrix with the coefficients w
     */
    auto *fft_w_coefficients = new complex<double>[2 * total_interpolation_points];
    auto *y_tilde_values = new double[total_interpolation_points * n_terms]();

    fftw_plan plan_dft, plan_idft;
    plan_dft = fftw_plan_dft_1d(2 * total_interpolation_points, reinterpret_cast<fftw_complex *>(fft_w_coefficients),
                                reinterpret_cast<fftw_complex *>(fft_w_coefficients), FFTW_FORWARD, FFTW_ESTIMATE);
    plan_idft = fftw_plan_dft_1d(2 * total_interpolation_points, reinterpret_cast<fftw_complex *>(fft_w_coefficients),
                                 reinterpret_cast<fftw_complex *>(fft_w_coefficients), FFTW_BACKWARD, FFTW_ESTIMATE);

    for (int d = 0; d < n_terms; d++) {
        for (int i = 0; i < 2 * total_interpolation_points; i++) {
            fft_w_coefficients[i].real(embedded_w_coefficients[i * n_terms + d]);
        }
        fftw_execute(plan_dft);

        // Take the Hadamard product of two complex vectors
        for (int i = 0; i < 2 * total_interpolation_points; i++) {
            double x_ = fft_w_coefficients[i].real();
            double y_ = fft_w_coefficients[i].imag();
            double u_ = fft_kernel_vector[i].real();
            double v_ = fft_kernel_vector[i].imag();
            fft_w_coefficients[i].real(x_ * u_ - y_ * v_);
            fft_w_coefficients[i].imag(x_ * v_ + y_ * u_);
        }

        // Invert the computed values at the interpolated nodes, unfortunate naming but it's better to do IDFT inplace
        fftw_execute(plan_idft);

        for (int i = 0; i < total_interpolation_points; i++) {
            // FFTW doesn't perform IDFT normalization, so we have to do it ourselves. This is done by multiplying the
            // result with the number of points in the input
            y_tilde_values[i * n_terms + d] = fft_w_coefficients[i].real() / (total_interpolation_points * 2.0);
        }
    }

    fftw_destroy_plan(plan_dft);
    fftw_destroy_plan(plan_idft);
    delete[] fft_w_coefficients;

    /*
     * Step 3: Compute the potentials \tilde{\phi}
     */
    for (int i = 0; i < N; i++) {
        int box_idx = point_box_idx[i] * n_interpolation_points;
        for (int j = 0; j < n_interpolation_points; j++) {
            for (int d = 0; d < n_terms; d++) {
                potentialsQij[i * n_terms + d] +=
                        interpolated_values[j * N + i] * y_tilde_values[(box_idx + j) * n_terms + d];
            }
        }
    }

    delete[] point_box_idx;
    delete[] y_in_box;
    delete[] interpolated_values;
    delete[] w_coefficients;
    delete[] y_tilde_values;
    delete[] embedded_w_coefficients;
}
