import numpy as np
from scipy.special import spherical_jn, spherical_yn, lpmv  # lpmv for vectorized Legendre polynomial computation
from numpy.polynomial.legendre import legval  # To compute the Legendre polynomial

def plane_wave_impinge_on_sphere(x, y, z, a=1.0, k=10.0, N=50):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) if r != 0 else 0
    
    n_values = np.arange(1, N+1)
    
    # Precompute spherical Bessel functions and their derivatives
    jn_values = spherical_jn(n_values, k*a)
    jn_prime_values = spherical_jn(n_values, k*a, derivative=True)
    yn_values = spherical_yn(n_values, k*a)
    yn_prime_values = spherical_yn(n_values, k*a, derivative=True)
    
    hn_prime_values = jn_prime_values + 1j * yn_prime_values
    
    # Precompute Legendre polynomials
    cos_theta = np.cos(theta)
    Pn_values = lpmv(0, n_values, cos_theta)
    
    # Precompute spherical Hankel function of the first kind for all n
    h_n_values = spherical_jn(n_values, k*r) + 1j * spherical_yn(n_values, k*r)
    
    # Compute the term for all n and sum them up
    terms = 1j**n_values * (2*n_values + 1) * (jn_prime_values / hn_prime_values) * Pn_values * h_n_values
    phi_s = np.sum(terms)
        
    return phi_s

x, y, z = 1.0, 1.0, 1.0
result = plane_wave_impinge_on_sphere(x, y, z)
print(result)