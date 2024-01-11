#include <stdio.h>
#include <stdlib.h>
#include <math.h>

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