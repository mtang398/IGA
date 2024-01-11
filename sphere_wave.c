#include <stdio.h>
#include <gsl/gsl_sf_bessel.h>
#include <gsl/gsl_sf_legendre.h>
#include <math.h>
#include <complex.h>

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

int main() {
    double complex result = plane_wave_impinge_on_sphere(1, 1, 1, 1.0, 10.0, 50);
    printf("Result: %f + %fi\n", creal(result), cimag(result));
    return 0;
}
