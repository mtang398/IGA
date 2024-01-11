#include <stdio.h>
#include <gsl/gsl_integration.h>
#include <gsl/gsl_matrix.h>

// Function to integrate
double func(double u, double v) {
    return u * u + v * v;
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

// Main function
int main() {
    double u_min = 0, u_max = 1, v_min = 0, v_max = 1;
    int n_points = 20;
    double result = gaussian_quadrature_2D(func, u_min, u_max, v_min, v_max, n_points);
    printf("Integral: %f\n", result);
    return 0;
}
