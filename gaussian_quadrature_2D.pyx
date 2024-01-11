import numpy as np
cimport numpy as np

def gaussian_quadrature_2D(func, int n_points=10):
    cdef np.ndarray[np.float64_t, ndim=1] points, weights
    cdef np.ndarray[np.float64_t, ndim=2] u_vals, v_vals, w_u, w_v, func_vals
    cdef np.float64_t integral
    
    # Define Gauss quadrature points and weights for interval [0, 1]
    points, weights = np.polynomial.legendre.leggauss(n_points)
    points = 0.5 * (points + 1)  # Map from [-1, 1] to [0, 1]
    weights *= 0.5  # Adjust weights for interval [0, 1]
    
    # Generate 2D grid of points and weights
    u_vals, v_vals = np.meshgrid(points, points)
    w_u, w_v = np.meshgrid(weights, weights)
    
    # Calculate function values for the entire grid
    func_vals = np.vectorize(func)(u_vals, v_vals)
    
    # Calculate the integral using vectorized operations
    integral = np.sum(func_vals * w_u * w_v)
    
    return integral