# Importing necessary libraries and functions
import numpy as np
cimport numpy as np  # This line is important to leverage Numpy's C-API

# Defining function with Cython, cpdef is used to create functions that can be called both from Cython and Python
cpdef compute_A_and_b_spline_cython(int n, object tensor_bspline_basis, object gaussian_quadrature_2D, 
                                    object surface_point_cffi, object cp, object w, object knot_vector_u_bottom, 
                                    object knot_vector_v_bottom, object random_function_space, 
                                    tuple u_interval, tuple v_interval, object knot_vector_u, 
                                    object knot_vector_v, int p, int q):
    
    # Declare variable types
    cdef int i, j, k, l, range_val
    cdef int size = (n + 2**k_ref) ** 2  # assuming k_ref is defined
    cdef double[:, :] A = np.zeros((size, size), dtype=np.float64)  # 2D Numpy array
    cdef double[:] b = np.zeros(size, dtype=np.float64)  # 1D Numpy array
    
    range_val = n + 2**k_ref
    
    # Iterate over the range and compute elements of A and b arrays
    for i in range(range_val):
        for j in range(range_val):
            idx_ij = i * range_val + j  # precomputing index
            
            for k in range(range_val):
                for l in range(range_val):
                    idx_kl = k * range_val + l  # precomputing index
                    
                    def integrand_A(double u, double v):
                        # Assuming tensor_bspline_basis can be called from Cython. 
                        # If it's a Python function, you might need to use cpdef for it as well.
                        return (tensor_bspline_basis(i, j, p, q, u, v, knot_vector_u, knot_vector_v) * 
                                tensor_bspline_basis(k, l, p, q, u, v, knot_vector_u, knot_vector_v))
                    
                    A[idx_ij, idx_kl] = gaussian_quadrature_2D(integrand_A, u_interval, v_interval)
                    
                    def integrand_b(double u, double v):
                        x, y, z = surface_point_cffi(u, v, cp, w, knot_vector_u_bottom, knot_vector_v_bottom)
                        return random_function_space(x, y, z) * tensor_bspline_basis(k, l, p, q, u, v, knot_vector_u, knot_vector_v)
                    
                    b[idx_kl] = gaussian_quadrature_2D(integrand_b, u_interval, v_interval)
    
    return A, b, n
