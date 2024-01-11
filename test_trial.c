#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_legendre.h>
#include <complex.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix.h>

#define CACHE_SIZE 1000 // Assuming a max size for demonstration

typedef struct {
    int i;
    int k;
    double t;
    double value;
} CacheItem;

double N(int i, int k, double t, double knot_vector[], int knot_vector_size, CacheItem cache[], int *cache_count) {
    // Check cache
    for (int c = 0; c < *cache_count; ++c) {
        if (cache[c].i == i && cache[c].k == k && cache[c].t == t) {
            return cache[c].value;
        }
    }

    double result = 0.0;

    if (t == knot_vector[knot_vector_size - 1] && i == knot_vector_size - k - 2) {
        result = 1.0;
    } else if (t == knot_vector[0] && i == 0) {
        result = 1.0;
    } else if (k == 0) {
        result = (knot_vector[i] <= t && t < knot_vector[i + 1]) ? 1.0 : 0.0;
    } else {
        double N1 = 0.0;
        double N2 = 0.0;
        if (knot_vector[i + k] - knot_vector[i] != 0) {
            N1 = ((t - knot_vector[i]) / (knot_vector[i + k] - knot_vector[i])) * N(i, k - 1, t, knot_vector, knot_vector_size, cache, cache_count);
        }
        if (knot_vector[i + k + 1] - knot_vector[i + 1] != 0) {
            N2 = ((knot_vector[i + k + 1] - t) / (knot_vector[i + k + 1] - knot_vector[i + 1])) * N(i + 1, k - 1, t, knot_vector, knot_vector_size, cache, cache_count);
        }
        result = N1 + N2;
    }

    // Add to cache
    if (*cache_count < CACHE_SIZE) {
        cache[*cache_count].i = i;
        cache[*cache_count].k = k;
        cache[*cache_count].t = t;
        cache[*cache_count].value = result;
        (*cache_count)++;
    }

    return result;
}

double R(int i, int j, int k, int l, double u, double v, double *knot_vector_u, int knot_vector_u_size, double *knot_vector_v, int knot_vector_v_size, double *weights, int weights_size, CacheItem *cache_u, int *cache_count_u, CacheItem *cache_v, int *cache_count_v) {
    double *N_u = (double *)malloc((knot_vector_u_size - k - 1) * sizeof(double));
    double *N_v = (double *)malloc((knot_vector_v_size - l - 1) * sizeof(double));

    for (int i_p = 0; i_p < knot_vector_u_size - k - 1; ++i_p) {
        N_u[i_p] = N(i_p, k, u, knot_vector_u, knot_vector_u_size, cache_u, cache_count_u);
    }
    for (int j_p = 0; j_p < knot_vector_v_size - l - 1; ++j_p) {
        N_v[j_p] = N(j_p, l, v, knot_vector_v, knot_vector_v_size, cache_v, cache_count_v);
    }

    double W_ij = 0.0;
    for (int i_p = 0; i_p < knot_vector_u_size - k - 1; ++i_p) {
        for (int j_p = 0; j_p < knot_vector_v_size - l - 1; ++j_p) {
            W_ij += N_u[i_p] * N_v[j_p] * weights[i_p * (knot_vector_v_size - l - 1) + j_p];
        }
    }

    free(N_u);
    free(N_v);

    return (N_u[i] * N_v[j] * weights[i * (knot_vector_v_size - l - 1) + j]) / W_ij;
}

void surface_point(double u, double v, double *control_points, int control_points_rows, int control_points_cols, double *weights, int weights_size, double *knot_vector_u, int knot_vector_u_size, double *knot_vector_v, int knot_vector_v_size, double point[3]) {
    int num_points_u = control_points_rows;
    int num_points_v = control_points_cols;
    int k = knot_vector_u_size - num_points_u - 1;
    int l = knot_vector_v_size - num_points_v - 1;

    point[0] = point[1] = point[2] = 0.0;

    CacheItem *cache_u = (CacheItem *)malloc(CACHE_SIZE * sizeof(CacheItem));
    CacheItem *cache_v = (CacheItem *)malloc(CACHE_SIZE * sizeof(CacheItem));
    int cache_count_u = 0;
    int cache_count_v = 0;

    for (int i = 0; i < num_points_u; ++i) {
        for (int j = 0; j < num_points_v; ++j) {
            double Rij = R(i, j, k, l, u, v, knot_vector_u, knot_vector_u_size, knot_vector_v, knot_vector_v_size, weights, weights_size, cache_u, &cache_count_u, cache_v, &cache_count_v);
            point[0] += Rij * control_points[(i * num_points_v + j) * 3 + 0];
            point[1] += Rij * control_points[(i * num_points_v + j) * 3 + 1];
            point[2] += Rij * control_points[(i * num_points_v + j) * 3 + 2];
        }
    }

    free(cache_u);
    free(cache_v);
}

// Function to calculate factorial
unsigned long factorial(int n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    unsigned long result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// Function to calculate binomial coefficient
unsigned long comb(int n, int k) {
    return factorial(n) / (factorial(k) * factorial(n - k));
}

// Function to compute the Bernstein basis
void bernstein_basis(int i, int n, double *u_array, double *result, int array_size) {
    unsigned long binomial_coeff = comb(n, i);
    for (int j = 0; j < array_size; ++j) {
        result[j] = binomial_coeff * pow(u_array[j], i) * pow(1 - u_array[j], n - i);
    }
}

// Function to compute the tensor Bernstein basis
void tensor_bernstein_basis(int i, int j, int n, double *u_array, double *v_array, double **result, int u_size, int v_size) {
    double *bernstein_u = (double *) malloc(u_size * sizeof(double));
    double *bernstein_v = (double *) malloc(v_size * sizeof(double));

    bernstein_basis(i, n, u_array, bernstein_u, u_size);
    bernstein_basis(j, n, v_array, bernstein_v, v_size);

    for (int u = 0; u < u_size; ++u) {
        for (int v = 0; v < v_size; ++v) {
            result[u][v] = bernstein_u[u] * bernstein_v[v];
        }
    }

    free(bernstein_u);
    free(bernstein_v);
}

double complex plane_wave_impinge_on_sphere(double x, double y, double z, double a, double k, int N) {
    double r = sqrt(x*x + y*y + z*z);
    double theta = r != 0 ? acos(z / r) : 0;
    
    double complex phi_s = 0;
    double h = 1e-5; // Small value for central difference
    double jn, yn, jn_prime, yn_prime, Pn;
    double complex hn_prime, h_n;

    for (int n = 1; n <= N; n++) {
        jn = gsl_sf_bessel_jl(n, k * a);
        yn = gsl_sf_bessel_yl(n, k * a);
        
        // Using central difference for derivatives
        jn_prime = (gsl_sf_bessel_jl(n, k * a + h) - gsl_sf_bessel_jl(n, k * a - h)) / (2 * h);
        yn_prime = (gsl_sf_bessel_yl(n, k * a + h) - gsl_sf_bessel_yl(n, k * a - h)) / (2 * h);

        hn_prime = jn_prime + I * yn_prime;

        // Compute the Legendre polynomials (special case of associated Legendre function for m=0)
        Pn = gsl_sf_legendre_Pl(n, cos(theta));

        h_n = gsl_sf_bessel_jl(n, k * r) + I * gsl_sf_bessel_yl(n, k * r);

        double complex term = cpow(I, n) * (2 * n + 1) * (jn_prime / hn_prime) * Pn * h_n;
        phi_s += term;
    }
    
    return phi_s;
}

// Gaussian quadrature function for 2D integration
double gaussian_quadrature_2D(double (*func)(double, double), double u_min, double u_max, double v_min, double v_max, int n_points) {
    gsl_integration_glfixed_table *table = gsl_integration_glfixed_table_alloc(n_points);

    double integral = 0.0;
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < n_points; ++j) {
            double u_point, v_point, w_u, w_v;

            gsl_integration_glfixed_point(u_min, u_max, i, &u_point, &w_u, table);
            gsl_integration_glfixed_point(v_min, v_max, j, &v_point, &w_v, table);

            integral += func(u_point, v_point) * w_u * w_v;
        }
    }

    gsl_integration_glfixed_table_free(table);
    return integral;
}

// Define a structure to represent a 3D array
typedef struct {
    double data[5][5][4];
} Matrix3D;

// Function to print a 3D matrix
void printMatrix(Matrix3D* matrix) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 4; k++) {
                printf("%f ", matrix->data[i][j][k]);
            }
            printf("\n");
        }
    }
}

// Define a function to slice a 3D array
void sliceArray(const Matrix3D* source, double target[5][5][3], int end_dim) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < end_dim; k++) {
                target[i][j][k] = source->data[i][j][k];
            }
        }
    }
}

// Define a function to compute actual control points
void computeActualControlPoints(double weightedControlPoints[5][5][3], double weights[5][5], double radius) {
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            double weight = weights[i][j];
            for (int k = 0; k < 3; k++) {
                weightedControlPoints[i][j][k] /= weight;
                weightedControlPoints[i][j][k] *= radius;
            }
        }
    }
}

typedef struct {
    int i, j, k, l, n;
    double **bernstein_results;
    double *knot_vector_u, *knot_vector_v;
    int knot_vector_u_size, knot_vector_v_size;
    double control_points[5][5][3];
    double weights[5][5];
} IntegrandsContext;

// Global variable for context
IntegrandsContext global_ctx; // Ensure this is declared globally

// Integrand for matrix A
double integrand_A(double u, double v, int i, int j, int k, int l, int n, double *knot_vector_u, int knot_vector_u_size, double *knot_vector_v, int knot_vector_v_size) {
    double **result_i_j = (double **)malloc((n+1) * sizeof(double *));
    double **result_k_l = (double **)malloc((n+1) * sizeof(double *));
    for (int idx = 0; idx < n+1; idx++) {
        result_i_j[idx] = (double *)malloc((n+1) * sizeof(double));
        result_k_l[idx] = (double *)malloc((n+1) * sizeof(double));
    }

    double u_array[1] = {u};
    double v_array[1] = {v};

    tensor_bernstein_basis(i, j, n, u_array, v_array, result_i_j, 1, 1);
    tensor_bernstein_basis(k, l, n, u_array, v_array, result_k_l, 1, 1);

    double product = 0.0;
    for (int u_idx = 0; u_idx <= n; u_idx++) {
        for (int v_idx = 0; v_idx <= n; v_idx++) {
            product += result_i_j[u_idx][v_idx] * result_k_l[u_idx][v_idx];
        }
    }

    for (int idx = 0; idx < n+1; idx++) {
        free(result_i_j[idx]);
        free(result_k_l[idx]);
    }
    free(result_i_j);
    free(result_k_l);

    return product;
}

// Integrand for vector b
double integrand_b(double u, double v, int k, int l, int n, double *knot_vector_u, int knot_vector_u_size, double *knot_vector_v, int knot_vector_v_size, double control_points[5][5][3], double weights[5][5]) {
    double point[3];
    surface_point(u, v, (double *)control_points, 5, 5, (double *)weights, 25, knot_vector_u, knot_vector_u_size, knot_vector_v, knot_vector_v_size, point);
    double complex f_value = plane_wave_impinge_on_sphere(point[0], point[1], point[2], 1.0, 1.0, n);

    double **result = (double **)malloc((n+1) * sizeof(double *));
    for (int idx = 0; idx < n+1; idx++) {
        result[idx] = (double *)malloc((n+1) * sizeof(double));
    }

    double u_array[1] = {u};
    double v_array[1] = {v};

    tensor_bernstein_basis(k, l, n, u_array, v_array, result, 1, 1);

    double product = 0.0;
    for (int u_idx = 0; u_idx <= n; u_idx++) {
        for (int v_idx = 0; v_idx <= n; v_idx++) {
            product += creal(f_value) * result[u_idx][v_idx];
        }
    }

    for (int idx = 0; idx < n+1; idx++) {
        free(result[idx]);
    }
    free(result);

    return product;
}

// Global variables for context
int global_i, global_j, global_k, global_l, global_n;
double **global_bernstein_results;
double global_control_points[5][5][3];
double global_weights[5][5];
double *global_knot_vector_u, *global_knot_vector_v;
int global_knot_vector_u_size, global_knot_vector_v_size;

void construct_matrices_A_and_b(double control_points[5][5][3], double weights[5][5], double knot_vector_u[], int knot_vector_u_size, double knot_vector_v[], int knot_vector_v_size, int n, double A[][n+1], double b[]) {
    // Set global variables
    memcpy(global_control_points, control_points, sizeof(global_control_points));
    memcpy(global_weights, weights, sizeof(global_weights));
    global_knot_vector_u = knot_vector_u;
    global_knot_vector_v = knot_vector_v;
    global_knot_vector_u_size = knot_vector_u_size;
    global_knot_vector_v_size = knot_vector_v_size;
    global_n = n;

    // Allocate memory for global_bernstein_results
    global_bernstein_results = (double **)malloc((n+1) * sizeof(double *));
    for (int idx = 0; idx <= n; idx++) {
        global_bernstein_results[idx] = (double *)malloc((n+1) * sizeof(double));
    }

    // Compute matrix A
    for (int i = 0; i <= n; i++) {
        for (int j = 0; j <= n; j++) {
            for (int k = 0; k <= n; k++) {
                for (int l = 0; l <= n; l++) {
                    global_i = i; global_j = j; global_k = k; global_l = l;
                    A[i*(n+1)+j][k*(n+1)+l] = gaussian_quadrature_2D(
                        integrand_A, 0, 1, 0, 1, 10
                    );
                }
            }
        }
    }

    // Compute vector b
    for (int k = 0; k <= n; k++) {
        for (int l = 0; l <= n; l++) {
            global_k = k; global_l = l;
            b[k*(n+1)+l] = gaussian_quadrature_2D(
                integrand_b, 0, 1, 0, 1, 10
            );
        }
    }

    // Free resources
    for (int idx = 0; idx <= n; idx++) {
        free(global_bernstein_results[idx]);
    }
    free(global_bernstein_results);
}


int main() {

    int p = 4;             // Order
    int k_ref = 0;
    int h_ref = 0;
    int h_check = 0;
    int n_refinement = 0;  // Level of refinement, can be set to any value between 0 and 8
    
    Matrix3D cp_data = {
        {
            {
                {4*(1 - sqrt(3)), 4*(1 - sqrt(3)), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))},
                {-sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)},
                {0, 4*(1 - 2*sqrt(3))/3, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3},
                {sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)},
                {4*(sqrt(3) - 1), 4*(1 - sqrt(3)), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))}
            },
            {
                {sqrt(2)*(sqrt(3) - 4), -sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)},
                {(2 - 3*sqrt(3))/2, (2 - 3*sqrt(3))/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2},
                {0, sqrt(2)*(2*sqrt(3) - 7)/3, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3},
                {(3*sqrt(3) - 2)/2, (2 - 3*sqrt(3))/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2},
                {sqrt(2)*(4 - sqrt(3)), -sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)}
            },
            {
                {4*(1 - 2*sqrt(3))/3, 0, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3},
                {sqrt(2)*(2*sqrt(3) - 7)/3, 0, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3},
                {0, 0, 4*(sqrt(3) - 5)/3, 4*(5*sqrt(3) - 1)/9},
                {sqrt(2)*(7 - 2*sqrt(3))/3, 0, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3},
                {4*(2*sqrt(3) - 1)/3, 0, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3}
            },
            {
                {sqrt(2)*(sqrt(3) - 4), sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)},
                {(2 - 3*sqrt(3))/2, (3*sqrt(3) - 2)/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2},
                {0, sqrt(2)*(7 - 2*sqrt(3))/3, -5*sqrt(6)/3, sqrt(2)*(sqrt(3) + 6)/3},
                {(3*sqrt(3) - 2)/2, (3*sqrt(3) - 2)/2, -(sqrt(3) + 6)/2, (sqrt(3) + 6)/2},
                {sqrt(2)*(4 - sqrt(3)), sqrt(2), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)}
            },
            {
                {4*(1 - sqrt(3)), 4*(sqrt(3) - 1), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))},
                {-sqrt(2), sqrt(2)*(4 - sqrt(3)), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)},
                {0, 4*(2*sqrt(3) - 1)/3, 4*(1 - 2*sqrt(3))/3, 4*(5 - sqrt(3))/3},
                {sqrt(2), sqrt(2)*(4 - sqrt(3)), sqrt(2)*(sqrt(3) - 4), sqrt(2)*(3*sqrt(3) - 2)},
                {4*(sqrt(3) - 1), 4*(sqrt(3) - 1), 4*(1 - sqrt(3)), 4*(3 - sqrt(3))}
            }
        }
    };

    printMatrix(&cp_data);

    // Define and initialize the knot vectors
    double knot_vector_u_bottom[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};
    double knot_vector_v_bottom[] = {0, 0, 0, 0, 0, 1, 1, 1, 1, 1};

    // Slice the control points to get the bottom part (3 components)
    double control_points_bottom[5][5][3];
    sliceArray(&cp_data, control_points_bottom, 3);

    // Get the weights
    double weights[5][5];
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            weights[i][j] = cp_data.data[i][j][3];
        }
    }

    // Compute actual control points
    double radius = 1.0;
    computeActualControlPoints(control_points_bottom, weights, radius);

    // Print the weights
    printf("Weights:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            printf("%.6f ", weights[i][j]);
        }
        printf("\n");
    }

    // Print the actual control points
    printf("\nActual Control Points:\n");
    for (int i = 0; i < 5; i++) {
        for (int j = 0; j < 5; j++) {
            for (int k = 0; k < 3; k++) {
                printf("%.6f ", control_points_bottom[i][j][k]);
            }
            printf("\n");
        }
    }

    // Define degree of Bernstein polynomials (degree should be set based on the problem requirement)
    int degree = p;

    // Define and initialize matrices A and b
    int matrix_size = (degree + 1) * (degree + 1);
    double A[matrix_size][matrix_size];
    double b[matrix_size];

    // Initialize A and b to zeros (if needed)
    for (int i = 0; i < matrix_size; i++) {
        b[i] = 0.0;
        for (int j = 0; j < matrix_size; j++) {
            A[i][j] = 0.0;
        }
    }

    // Call the function to construct matrices A and b
    construct_matrices_A_and_b(control_points_bottom, weights, knot_vector_u_bottom, sizeof(knot_vector_u_bottom) / sizeof(knot_vector_u_bottom[0]), knot_vector_v_bottom, sizeof(knot_vector_v_bottom) / sizeof(knot_vector_v_bottom[0]), degree, A, b);

    // Print matrices A and b (optional, for verification)
    printf("Matrix A:\n");
    for (int i = 0; i < matrix_size; i++) {
        for (int j = 0; j < matrix_size; j++) {
            printf("%.6f ", A[i][j]);
        }
        printf("\n");
    }

    printf("\nVector b:\n");
    for (int i = 0; i < matrix_size; i++) {
        printf("%.6f ", b[i]);
    }
    printf("\n");

    return 0;
}